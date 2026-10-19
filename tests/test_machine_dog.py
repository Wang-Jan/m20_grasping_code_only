import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp.machine_dog import ApproachResult, M20Config, M20DogAdapter
from m20_rebot_moving_grasp.machine_dog.m20_protocol import (
    UdpJsonTransport, build_protocol_header, perception_state_request, single_point_nav_request,
)


class FakeSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
    def bind(self, address): pass
    def settimeout(self, timeout): pass
    def sendto(self, packet, address): self.sent.append(packet)
    def recvfrom(self, size): return self.responses.pop(0)
    def close(self): pass


class ProtocolTest(unittest.TestCase):
    def test_header_and_message_id_filter(self):
        fake = FakeSocket([
            (build_protocol_header(2, 77) + b"{}", ("x", 1)),
            (build_protocol_header(2, 1) + b"{}", ("x", 1)),
        ])
        transport = UdpJsonTransport("10.21.31.103", socket_factory=lambda *args: fake)
        self.assertEqual(transport.send_request("{}"), {})
        self.assertEqual(int.from_bytes(fake.sent[0][6:8], "little"), 1)

    def test_navigation_compat_can_accept_mismatched_message_id(self):
        fake = FakeSocket([
            (build_protocol_header(2, 77) + b"{}", ("x", 1)),
        ])
        transport = UdpJsonTransport("10.21.31.103", socket_factory=lambda *args: fake)
        self.assertEqual(transport.send_request("{}", allow_response_id_mismatch=True), {})

    def test_header_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            build_protocol_header(70000, 1)

    def test_perception_and_dynamic_navigation_requests(self):
        perception = json.loads(perception_state_request())
        self.assertEqual(perception["PatrolDevice"]["Type"], 2002)
        self.assertEqual(perception["PatrolDevice"]["Command"], 1)
        payload = json.loads(single_point_nav_request(
            {"point_id": 901, "x": 1.2, "y": -0.5, "z": 0.0, "yaw": 0.3},
            gait=0x3002, speed=1, obs_mode=0, nav_mode=0,
        ))
        items = payload["PatrolDevice"]["Items"]
        self.assertEqual(payload["PatrolDevice"]["Time"], "2023-01-01 00:00:00")
        self.assertEqual((items["Value"], items["Gait"], items["Speed"], items["ObsMode"], items["NavMode"]),
                         (901, 0x3002, 1, 0, 0))


class AdapterTest(unittest.TestCase):
    def config(self, points):
        temp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(points, temp); temp.close()
        return M20Config(points_file=temp.name, clean_scan_sec=0.01, nav_timeout_sec=0.2).validate()

    def test_named_points_are_required_and_not_guessed(self):
        adapter = M20DogAdapter(self.config({"point0": {"x": 0, "y": 0, "z": 0, "yaw": 0}}), transport=object())
        self.assertFalse(adapter._load_point("scan_point"))

    def test_static_navigation_uses_coordinates_without_saved_point_id(self):
        cfg = self.config({"scan_point": {"x": 1, "y": 2, "z": 0, "yaw": 0}})
        class Transport:
            def __init__(self):
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0, "Location": 0}}},
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                    {"PatrolDevice": {"Items": {"Value": 0, "Status": 4, "ErrorCode": 0}}},
                ]
                self.payloads = []
                self.timeouts = []
            def send_request(self, payload, timeout):
                self.payloads.append(json.loads(payload))
                self.timeouts.append(timeout)
                return self.responses.pop(0)
        transport = Transport()
        self.assertTrue(M20DogAdapter(cfg, transport=transport).goto_target())
        items = transport.payloads[1]["PatrolDevice"]["Items"]
        self.assertEqual(items["Value"], 0)
        self.assertEqual((items["PosX"], items["PosY"], items["PosZ"], items["AngleYaw"]), (1.0, 2.0, 0.0, 0.0))
        self.assertEqual(transport.timeouts[1], cfg.nav_timeout_sec)

    def test_static_navigation_accepts_immediate_completed_response(self):
        cfg = self.config({"scan_point": {"x": 1, "y": 2, "z": 0, "yaw": 0}})
        class Transport:
            def __init__(self):
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0, "Location": 0}}},
                    {"PatrolDevice": {"Items": {"Status": 0, "ErrorCode": 0x2300}}},
                ]
                self.calls = 0
            def send_request(self, payload, timeout):
                self.calls += 1
                return self.responses.pop(0)
        transport = Transport()
        self.assertTrue(M20DogAdapter(cfg, transport=transport).goto_target())
        self.assertEqual(transport.calls, 2)

    def test_approach_accepts_only_confirmed_prone_result(self):
        cfg = self.config({})
        for result, expected in ((ApproachResult.PRONE_AT_TARGET, True), (ApproachResult.NO_TARGET, False),
                                 (ApproachResult.TIMEOUT, False), (True, False), (False, False)):
            adapter = M20DogAdapter(cfg, transport=object(), approach_handler=lambda result=result: result)
            self.assertEqual(adapter.approach(), expected)

    def test_clean_scan_returns_false_when_target_is_seen(self):
        cfg = self.config({})
        calls = [[], ["bottle"]]
        adapter = M20DogAdapter(cfg, transport=object(), scan_handler=lambda: calls.pop(0) if calls else [])
        with patch("m20_rebot_moving_grasp.machine_dog.m20_adapter.time.sleep", return_value=None):
            self.assertFalse(adapter.is_clean())

    def test_clean_scan_returns_true_after_no_target_window(self):
        cfg = self.config({})
        adapter = M20DogAdapter(cfg, transport=object(), scan_handler=lambda: [])
        with patch("m20_rebot_moving_grasp.machine_dog.m20_adapter.time.sleep", return_value=None):
            self.assertTrue(adapter.is_clean())

    def test_scan_stop_is_deferred_until_approach(self):
        close_calls = []
        cfg = self.config({})
        adapter = M20DogAdapter(
            cfg,
            transport=object(),
            scan_handler=lambda: ["bottle"],
            scan_start=lambda: None,
            scan_stop=lambda: close_calls.append("closed"),
            approach_handler=lambda: ApproachResult.PRONE_AT_TARGET,
        )
        with patch("m20_rebot_moving_grasp.machine_dog.m20_adapter.time.sleep", return_value=None):
            self.assertFalse(adapter.is_clean())
        self.assertEqual(close_calls, [])
        self.assertTrue(adapter.approach())
        self.assertEqual(close_calls, ["closed"])

    def test_navigation_uses_only_configured_names(self):
        cfg = self.config({"scan_point": {"point_id": 7, "x": 1, "y": 2, "z": 0, "yaw": 0}})
        class Transport:
            def __init__(self): self.responses = [
                {"PatrolDevice": {"Items": {"ErrorCode": 0, "Location": 0}}},
                {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                {"PatrolDevice": {"Items": {"Value": 7, "Status": 4, "ErrorCode": 0}}},
            ]
            def send_request(self, payload, timeout): return self.responses.pop(0)
        self.assertTrue(M20DogAdapter(cfg, transport=Transport()).goto_target())

    def test_motion_commands_parse_acknowledgement(self):
        cfg = self.config({})
        class Transport:
            def __init__(self): self.responses = [
                {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
            ]
            def send_request(self, payload, timeout): return self.responses.pop(0)
        adapter = M20DogAdapter(cfg, transport=Transport())
        self.assertTrue(adapter.stand())
        self.assertTrue(adapter.sit())
        self.assertTrue(adapter._motion(2))

    def test_program_abort_cancels_navigation_then_lies_instead_of_soft_stopping(self):
        cfg = self.config({})

        class Transport:
            def __init__(self):
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                    {"PatrolDevice": {"Items": {"ErrorCode": 0x2302}}},
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                ]
                self.payloads = []
            def send_request(self, payload, timeout):
                self.payloads.append(json.loads(payload))
                return self.responses.pop(0)

        transport = Transport()
        self.assertTrue(M20DogAdapter(cfg, transport=transport).emergency_stop())
        requests = [payload["PatrolDevice"] for payload in transport.payloads]
        self.assertEqual([(request["Type"], request["Command"]) for request in requests], [(1004, 1), (1007, 1), (2, 22)])
        self.assertEqual(requests[-1]["Items"]["MotionParam"], 4)

    def test_cancel_accepts_completed_navigation_without_waiting(self):
        cfg = self.config({})

        class Transport:
            def __init__(self):
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                    {"PatrolDevice": {"Items": {"Value": 902, "Status": 0, "ErrorCode": 0x2300}}},
                ]
                self.payloads = []

            def send_request(self, payload, timeout):
                self.payloads.append(json.loads(payload))
                return self.responses.pop(0)

        transport = Transport()
        self.assertTrue(M20DogAdapter(cfg, transport=transport)._cancel_navigation())
        self.assertEqual(len(transport.payloads), 2)


if __name__ == "__main__":
    unittest.main()
