import time
import threading

_log_buffer = []
_log_lock   = threading.Lock()
_last_log_time: dict = {}


def add_log(track_id, name, helmet, gloves, safety_vest, fallen):
    now        = time.time()
    person_key = f"{track_id}:{name}"

    with _log_lock:
        if now - _last_log_time.get(person_key, 0) < 1.0:
            return
        _last_log_time[person_key] = now

        _log_buffer.insert(0, {
            "id":           int(now * 1000),
            "track_id":     track_id,
            "employeeName": name,
            "helmet":       "yes" if helmet      else "no",
            "glove":        "yes" if gloves      else "no",
            "safety_vest":  "yes" if safety_vest else "no",
            "fallen":       "yes" if fallen      else "no",
            "time":         time.strftime("%H:%M:%S"),
        })
        _log_buffer[:] = _log_buffer[:200]


def get_logs():
    with _log_lock:
        return list(_log_buffer)


def reset_logs():
    global _log_buffer, _last_log_time
    with _log_lock:
        _log_buffer    = []
        _last_log_time = {}
