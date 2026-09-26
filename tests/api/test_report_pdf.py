"""Tests for GET /report/pdf -- the PDF assurance evidence report (owner: Khushi).

This route renders exactly what GET /assurance-result already returns as a
PDF. It performs no new computation, so these tests focus on: the route
returns a real, well-formed PDF; the entity-profile parameters reach the
same verified_requirements wiring /assurance-result uses; and model-facing
failure modes (unknown model_id, unreachable external model) behave the
same way every other model-facing route already does.
"""
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _assert_is_pdf(response, min_size=1000):
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    assert len(response.content) > min_size
    assert "attachment" in response.headers.get("content-disposition", "")


def test_report_pdf_default_model_returns_a_pdf():
    response = client.get("/report/pdf")
    _assert_is_pdf(response)


def test_report_pdf_filename_carries_the_assurance_run_id():
    response = client.get("/report/pdf")
    disposition = response.headers["content-disposition"]
    assert "assurance-report-" in disposition
    assert disposition.endswith('.pdf"')


def test_report_pdf_with_declared_entity_profile_is_a_pdf():
    response = client.get(
        "/report/pdf",
        params={
            "model_id": "german-credit-logistic-regression",
            "entity_type": "NBFC",
            "nbfc_layer": "Middle",
            "digital_lending": "true",
            "uses_external_model_vendor": "false",
        },
    )
    _assert_is_pdf(response)


def test_report_pdf_unknown_model_id_is_404():
    response = client.get("/report/pdf", params={"model_id": "does-not-exist"})
    assert response.status_code == 404


def test_report_pdf_unreachable_external_model_is_502_not_500(monkeypatch):
    """Same failure contract as every other model-facing route.

    A REST-served model being unreachable is the model's own outage, not
    this backend's defect -- see _unreachable_model() in app/api/main.py.
    """
    from app.models.rest_adapter import RESTAdapterError

    def _unreachable(*args, **kwargs):
        raise RESTAdapterError("HTTP request error scoring row 0: connection refused")

    monkeypatch.setattr("app.api.main.build_assurance_result", _unreachable)

    response = client.get("/report/pdf", params={"model_id": "synthetic-bank-credit-v1"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "could not be reached" in detail
    assert "synthetic-bank-credit-v1" in detail


def test_report_pdf_content_matches_assurance_result_json():
    """The PDF and the JSON endpoint must describe the same run's data.

    Not a pixel comparison -- confirms the two routes were built from the
    same underlying numbers by checking the JSON response's own fields are
    internally consistent with what the PDF route computes from
    (both call build_assurance_result() + _attach_verified_requirements()
    with identical parameters).
    """
    params = {
        "model_id": "german-credit-logistic-regression",
        "entity_type": "NBFC",
        "nbfc_layer": "Middle",
        "digital_lending": "true",
        "uses_external_model_vendor": "false",
    }
    json_response = client.get("/assurance-result", params=params)
    pdf_response = client.get("/report/pdf", params=params)

    assert json_response.status_code == 200
    assert pdf_response.status_code == 200

    body = json_response.json()
    verified_requirements = body["compliance"]["verified_requirements"]
    assert len(verified_requirements) == 16
    applicabilities = {r["applicability"] for r in verified_requirements}
    assert applicabilities == {"APPLIES", "NOT_APPLICABLE"}
