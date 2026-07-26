"""29.3 pagination tests."""

from polymarket.collection.pagination import paginate_cursor, paginate_offset


def _paged_fetch(pages):
    def fetch(params):
        offset = params["offset"]
        limit = params["limit"]
        idx = offset // limit
        records = pages[idx] if idx < len(pages) else []
        return idx + 1, records
    return fetch


def test_single_page():
    out = paginate_offset(_paged_fetch([[1, 2]]), limit=5)
    assert out.status == "complete"
    assert out.record_count == 2
    assert len(out.pages) == 1


def test_multiple_pages_and_empty_final_page():
    pages = [[1, 2, 3], [4, 5, 6], []]
    out = paginate_offset(_paged_fetch(pages), limit=3)
    assert out.status == "complete"
    assert out.record_count == 6
    assert len(out.pages) == 3


def test_repeated_page_detection():
    def fetch(params):
        return 1, [1, 2, 3]  # same full page forever

    out = paginate_offset(fetch, limit=3, max_pages=10)
    assert out.status == "incomplete"
    assert "repeated page" in out.note


def test_max_page_protection():
    def fetch(params):
        return 1, list(range(params["offset"], params["offset"] + 3))

    out = paginate_offset(fetch, limit=3, max_pages=4)
    assert out.status == "incomplete"
    assert "max_pages" in out.note
    assert out.next_state == {"offset": 12}


def test_interruption_and_resume():
    calls = {"fail_once": True}
    pages = [[1, 2, 3], [4, 5, 6], [7]]

    def fetch(params):
        idx = params["offset"] // 3
        if idx == 1 and calls["fail_once"]:
            calls["fail_once"] = False
            return -1, None
        return idx, pages[idx] if idx < len(pages) else []

    first = paginate_offset(fetch, limit=3)
    assert first.status == "failed"
    resumed = paginate_offset(
        fetch, limit=3, start_offset=first.next_state["offset"]
    )
    assert resumed.status == "complete"
    assert resumed.record_count == 4


def test_cursor_pagination_with_repeat_detection():
    def fetch(params):
        cursor = params.get("next_cursor")
        if cursor is None:
            return 1, [{"v": 1, "next": "a"}]
        return 1, [{"v": 2, "next": "a"}]  # repeats cursor "a"

    out = paginate_cursor(
        fetch, extract_cursor=lambda recs: recs[-1]["next"] if recs else None
    )
    assert out.status == "incomplete"
    assert "repeated cursor" in out.note
