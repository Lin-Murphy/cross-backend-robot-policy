"""Passive LeRobot bus trace: record raw writes without issuing extra bus operations."""
from functools import wraps
import time


def install_legacy_bus_audit(bus, record, clock=time.perf_counter_ns, goal_guard=None):
    """Wrap this bus instance's existing low-level writes and return a restore callback.

    The trace sees transmit attempts/return codes, not per-servo acknowledgements.
    It never reads registers, changes values, or retries a command.
    """
    originals={}
    for method in ('_write','_sync_write'):
        original=getattr(bus,method)
        originals[method]=original
        @wraps(original)
        def traced(*args,_method=method,_original=original,**kwargs):
            start=clock()
            event={'event':'legacy_bus_transmit','method':_method,'host_start_ns':start,
                   'args':list(args),'kwargs':kwargs,'attempted':True,'returned':False}
            checked=None
            if _method=='_sync_write' and goal_guard is not None and len(args)>=3 and args[0]==42:
                checked=goal_guard.check_goal_packet(args[0],args[1],args[2])
            if _method=='_write' and goal_guard is not None and args and args[0]==42:
                goal_guard.reject(['individual_goal_write_forbidden'])
            record(event)
            try:
                result=_original(*args,**kwargs)
            except BaseException as exc:
                record({'event':'legacy_bus_transmit_error','method':_method,
                        'host_end_ns':clock(),'error_type':type(exc).__name__,'error':str(exc)})
                raise
            record({'event':'legacy_bus_transmit_return','method':_method,
                    'host_end_ns':clock(),'return_value':result})
            if checked is not None:goal_guard.accepted_transmit(*checked)
            return result
        setattr(bus,method,traced)
    def restore():
        for method,original in originals.items():setattr(bus,method,original)
    return restore
