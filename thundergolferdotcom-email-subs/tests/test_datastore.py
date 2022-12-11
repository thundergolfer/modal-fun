from email_subs import datastore


def test_create_sub(store):
    actual = store.create_sub(email="foo@gmail.com")
    expected = datastore.Subscriber(
        email="foo@gmail.com",
        confirm_code="1234abcd",
        unsub_code="1234abcd",
        confirmed=False,
        unsubbed=False,
        created_at=None,
        confirmed_at=None,
        unsubbed_at=None,
        referrer=""
    )
    assert expected == actual
