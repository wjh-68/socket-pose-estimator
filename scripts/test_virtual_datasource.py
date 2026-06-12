import threading
import time
from config.app_config import AppConfig
from data_source.factory import build_datasource


def main():
    cfg = AppConfig.from_yaml("config/online_test_virtual.yaml")
    stop_event = threading.Event()
    ds = build_datasource(cfg.data_source, stop_event)
    try:
        ok = ds.initialize()
        print("initialize:", ok)
        ds.start()
        # read a few packets
        for i in range(10):
            pkt = ds.get_packet()
            print(f"packet {i}:", "ok" if pkt is not None else "None")
            time.sleep(0.05)
    finally:
        stop_event.set()
        ds.stop()
        ds.cleanup()


if __name__ == "__main__":
    main()
