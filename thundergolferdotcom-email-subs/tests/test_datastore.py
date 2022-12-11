import pytz
from datetime import datetime
import pytest

from email_subs import datastore


def test_create_sub(store):
    test_email = "foo@gmail.com"
    actual = store.create_sub(email=test_email)
    expected = datastore.Subscriber(
        email=test_email,
        confirm_code="1234abcd",
        unsub_code="1234abcd",
        confirmed=False,
        unsubbed=False,
        created_at=datetime.fromtimestamp(0, tz=pytz.utc),
        confirmed_at=None,
        unsubbed_at=None,
        referrer="",
    )
    assert expected == actual


def test_confirm_sub(store):
    test_email = "foo@gmail.com"
    sub = store.create_sub(email=test_email)

    assert not sub.confirmed
    assert not sub.confirmed_at

    confirmed = store.confirm_sub(
        email=test_email,
        code=sub.confirm_code,
    )
    assert confirmed

    sub = store.get_sub(test_email)

    assert sub.confirmed
    assert sub.confirmed_at


def test_confirm_sub_with_bad_code(store):
    test_email = "me@gmail.com"
    sub = store.create_sub(email=test_email)

    assert not sub.confirmed
    assert not sub.confirmed_at

    with pytest.raises(ValueError):
        store.confirm_sub(
            email=test_email,
            code="garbage-code-this-should-not-work",
        )

    assert not sub.confirmed
    assert not sub.confirmed_at


def test_unsub(store):
    test_email = "foo@gmail.com"
    sub = store.create_sub(email=test_email)
    assert store.confirm_sub(
        email=test_email,
        code=sub.confirm_code,
    )

    assert not sub.unsubbed
    assert not sub.unsubbed_at

    with pytest.raises(ValueError):
        store.unsub(
            email=test_email,
            code="garbage-code-this-should-not-work",
        )

    assert not sub.unsubbed
    assert not sub.unsubbed_at

    unsubbed = store.unsub(
        email=test_email,
        code=sub.unsub_code,
    )
    assert unsubbed

    sub = store.get_sub(test_email)

    assert sub.unsubbed
    assert sub.unsubbed_at
