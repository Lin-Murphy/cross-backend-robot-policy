/* Camera-only V4L2 probe. No S_FMT/S_PARM/control writes, no serial access.
 * Retain buffer timestamp semantics; SOE flag is driver evidence, not optical validation.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define FRAMES 60
#define BUFS 4
struct camera { const char *name, *path; int fd, active, count; unsigned nbuf;
    void *mem[BUFS]; size_t length[BUFS]; FILE *raw; };
static int64_t now_ns(void) { struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) return -1;
    return (int64_t)t.tv_sec*1000000000LL+t.tv_nsec; }
static int xioctl(int fd, unsigned long req, void *arg) { int r; do {r=ioctl(fd,req,arg);} while(r<0&&errno==EINTR); return r; }
static int start(struct camera *c, const char *dir) {
    c->fd=open(c->path,O_RDWR|O_NONBLOCK); if(c->fd<0) return -1;
    struct v4l2_format fmt={.type=V4L2_BUF_TYPE_VIDEO_CAPTURE};
    if(xioctl(c->fd,VIDIOC_G_FMT,&fmt)<0) return -1;
    if(fmt.fmt.pix.width!=640||fmt.fmt.pix.height!=480||fmt.fmt.pix.pixelformat!=V4L2_PIX_FMT_MJPEG) {errno=EINVAL;return -1;}
    char path[4096]; if(snprintf(path,sizeof path,"%s/%s.mjpg",dir,c->name)>=(int)sizeof path) {errno=ENAMETOOLONG;return -1;}
    c->raw=fopen(path,"wx");if(!c->raw)return -1;
    struct v4l2_requestbuffers req={.count=BUFS,.type=V4L2_BUF_TYPE_VIDEO_CAPTURE,.memory=V4L2_MEMORY_MMAP};
    if(xioctl(c->fd,VIDIOC_REQBUFS,&req)<0)return -1;
    if(req.count<2||req.count>BUFS){errno=EINVAL;return -1;} c->nbuf=req.count;
    for(unsigned i=0;i<c->nbuf;i++) {
        struct v4l2_buffer b={.type=V4L2_BUF_TYPE_VIDEO_CAPTURE,.memory=V4L2_MEMORY_MMAP,.index=i};
        if(xioctl(c->fd,VIDIOC_QUERYBUF,&b)<0)return -1;
        c->length[i]=b.length;c->mem[i]=mmap(NULL,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,c->fd,b.m.offset);
        if(c->mem[i]==MAP_FAILED)return -1;
        if(xioctl(c->fd,VIDIOC_QBUF,&b)<0)return -1;
    }
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(xioctl(c->fd,VIDIOC_STREAMON,&type)<0)return -1;
    c->active=1;
    return 0;
}
static int stop(struct camera *c) {
    int rc=0;
    if(c->fd>=0) {
        if(c->active) {enum v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE;if(xioctl(c->fd,VIDIOC_STREAMOFF,&t)<0)rc=-1;}
        for(unsigned i=0;i<c->nbuf;i++)if(c->mem[i]&&c->mem[i]!=MAP_FAILED)if(munmap(c->mem[i],c->length[i]))rc=-1;
        if(close(c->fd))rc=-1;
    }
    if(c->raw&&fclose(c->raw))rc=-1;
    return rc;
}
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"usage: probe existing-output-directory\n");return 2;}
    struct camera cams[2]={
        {.name="follower",.path="/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0",.fd=-1},
        {.name="camera2",.path="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",.fd=-1}};
    int rc=1; int64_t begin=now_ns();
    for(int i=0;i<2;i++)if(start(&cams[i],argv[1])<0){perror("camera start");goto cleanup;}
    while(cams[0].count<FRAMES||cams[1].count<FRAMES) {
        if(now_ns()-begin>10000000000LL){fprintf(stderr,"probe deadline exceeded\n");goto cleanup;}
        struct pollfd p[2];
        for(int i=0;i<2;i++)p[i]=(struct pollfd){.fd=cams[i].count<FRAMES?cams[i].fd:-1,.events=POLLIN};
        int n=poll(p,2,1000);if(n<0){if(errno==EINTR)continue;perror("poll");goto cleanup;}if(!n)continue;
        for(int i=0;i<2;i++) {
            if(p[i].revents&(POLLERR|POLLHUP|POLLNVAL)){fprintf(stderr,"poll device error\n");goto cleanup;}
            if(!(p[i].revents&POLLIN))continue;
            struct camera *c=&cams[i];struct v4l2_buffer b={.type=V4L2_BUF_TYPE_VIDEO_CAPTURE,.memory=V4L2_MEMORY_MMAP};
            int64_t before=now_ns();
            if(xioctl(c->fd,VIDIOC_DQBUF,&b)<0){if(errno==EAGAIN)continue;perror("dequeue");goto cleanup;}
            int64_t after=now_ns();
            if(b.index>=c->nbuf||b.bytesused>c->length[b.index]){fprintf(stderr,"invalid buffer length/index\n");goto cleanup;}
            long pos=ftell(c->raw);if(pos<0||fwrite(c->mem[b.index],1,b.bytesused,c->raw)!=b.bytesused){perror("save frame");goto cleanup;}
            printf("{\"camera\":\"%s\",\"index\":%d,\"sequence\":%u,\"flags\":%u,\"driver_timestamp_ns\":%" PRId64 ",\"host_dequeue_start_ns\":%" PRId64 ",\"host_dequeue_end_ns\":%" PRId64 ",\"byte_offset\":%ld,\"bytesused\":%u,\"timestamp_monotonic\":%s,\"timestamp_source_soe\":%s,\"buffer_error\":%s,\"optically_validated\":false}\n",
                c->name,c->count,b.sequence,b.flags,(int64_t)((int64_t)b.timestamp.tv_sec*1000000000LL+(int64_t)b.timestamp.tv_usec*1000),
                before,after,pos,b.bytesused,
                (b.flags&V4L2_BUF_FLAG_TIMESTAMP_MASK)==V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC?"true":"false",
                (b.flags&V4L2_BUF_FLAG_TSTAMP_SRC_MASK)==V4L2_BUF_FLAG_TSTAMP_SRC_SOE?"true":"false",
                b.flags&V4L2_BUF_FLAG_ERROR?"true":"false");
            if(fflush(stdout)){perror("log");goto cleanup;}
            c->count++;
            if(xioctl(c->fd,VIDIOC_QBUF,&b)<0){perror("requeue");goto cleanup;}
        }
    }
    rc=0;
cleanup:
    for(int i=0;i<2;i++)if(stop(&cams[i])<0)rc=1;
    return rc;
}
