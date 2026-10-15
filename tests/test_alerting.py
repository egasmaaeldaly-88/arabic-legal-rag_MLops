from arabic_legal_rag import alerting


def test_ok_when_above_threshold(tmp_path):
    log = tmp_path / "alerts.log"
    assert alerting.check({"faithfulness": 0.9}, 0.80, None, log) == 0
    assert not log.exists()


def test_alert_below_threshold_logs_and_sends(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(alerting, "send_webhook", lambda url, msg: sent.append((url, msg)) or True)
    log = tmp_path / "alerts.log"
    code = alerting.check({"faithfulness": 0.70, "num_questions": 20}, 0.80, "https://example/hook", log)
    assert code == 2
    assert "ALERT" in log.read_text(encoding="utf-8")
    assert len(sent) == 1 and "0.700" in sent[0][1]


def test_alert_without_webhook_still_triggers(tmp_path):
    assert alerting.check({"faithfulness": 0.5}, 0.80, None, tmp_path / "a.log") == 2


def test_missing_score_is_error(tmp_path):
    assert alerting.check({}, 0.80, None, tmp_path / "a.log") == 1


def test_payload_shape_per_service():
    assert alerting.payload_for("https://discord.com/api/webhooks/x", "m") == {"content": "m"}
    assert alerting.payload_for("https://hooks.slack.com/services/x", "m") == {"text": "m"}
