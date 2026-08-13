"""Read-only bridge from audited SO101 raw packets to common action receipts."""
from dataclasses import asdict
from .legacy_so101_sync_guard import NAMES as JOINTS
from .execution_contract import BackendCapabilities, ActionRequest, ActionReceipt, StopReceipt

RAW_UNITS = ('raw_tick',) * 6


def bridge_legacy_receipts(events, expected_transmitted):
    """Transport returns are physical sends, never per-servo acknowledgements."""
    capabilities = BackendCapabilities('so101', JOINTS, RAW_UNITS,
        ('follower', 'camera2'), 'host_monotonic', 'manual', 'operator',
        'sync_transport_return', True).validate()
    receipts = []
    for event in events:
        kind = event.get('event')
        if kind == 'legacy_goal_packet_transmitted':
            raw = event['raw']
            if len(raw) != 6 or event.get('transport_returned') is not True or \
               event.get('per_motor_acknowledged') is not False:
                raise ValueError('invalid audited goal transport evidence')
            target = tuple(raw)
            request = ActionRequest('source_exposure_unverified', JOINTS, RAW_UNITS, target)
            receipt = ActionReceipt(True, target, 'sync_transport_return', 1,
                                    event['host_ns']).validate(request, capabilities)
            receipts.append(asdict(receipt))
        elif kind == 'legacy_goal_candidate' and event.get('reasons'):
            raw = event['raw_ids_values']
            if set(raw) != set(str(i) for i in range(1, 7)) or event.get('hardware_dispatched') is not False:
                raise ValueError('invalid rejected goal candidate evidence')
            target = tuple(raw[str(i)] for i in range(1, 7))
            request = ActionRequest('source_exposure_unverified', JOINTS, RAW_UNITS, target)
            receipt = ActionReceipt(False, target, 'sync_transport_return', 0,
                                    event['host_ns'], ';'.join(event['reasons'])).validate(request, capabilities)
            receipts.append(asdict(receipt))
    physical = sum(r['physical_dispatches'] for r in receipts)
    if physical != expected_transmitted:
        raise ValueError('physical transport count does not match run summary')
    return asdict(capabilities), receipts


def bridge_legacy_stop(events):
    """One audited current-position hold; transport return is not servo acknowledgement."""
    attempts = [e for e in events if e.get('event') == 'legacy_stop_hold_attempt']
    returns = [e for e in events if e.get('event') == 'legacy_stop_hold_transport_return']
    if len(attempts) != 1 or len(returns) != 1 or \
       type(attempts[0].get('host_ns')) is not int or type(returns[0].get('host_ns')) is not int or \
       returns[0]['host_ns'] < attempts[0]['host_ns'] or \
       returns[0].get('per_motor_acknowledged') is not False:
        raise ValueError('incomplete or invalid stop-hold transport evidence')
    capabilities = BackendCapabilities('so101', JOINTS, RAW_UNITS,
        ('follower', 'camera2'), 'host_monotonic', 'manual', 'operator',
        'sync_transport_return', True).validate()
    return asdict(StopReceipt(True, 'current_position_sync_hold', 1,
                              'sync_transport_return').validate(capabilities))
