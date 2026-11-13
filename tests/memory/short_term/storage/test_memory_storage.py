
import pytest

from hiyal.memory.short_term import ContextEntry, InMemoryStorage
from hiyal.prompts import UserPrompt

S = "session-1"


def context(text):
    return ContextEntry(text=text)


@pytest.fixture
def storage():
    return InMemoryStorage()


class TestInMemoryStorage:
    def test_starts_empty(self, storage):
        assert storage.entries(S) == []

    def test_keeps_entries_in_the_order_they_were_added(self, storage):
        entries = [context("a"), UserPrompt(text="b"), context("c")]
        for entry in entries:
            storage.add(S, entry)
        assert storage.entries(S) == entries

    def test_does_not_check_for_duplicates(self, storage):
        entry = context("a")
        storage.add(S, entry)
        storage.add(S, entry)
        assert len(storage.entries(S)) == 2

    def test_remove_keeps_the_order_of_the_rest(self, storage):
        a, b, c = context("a"), context("b"), context("c")
        for entry in (a, b, c):
            storage.add(S, entry)
        storage.remove(S, b.id_)
        assert storage.entries(S) == [a, c]

    def test_removing_an_entry_that_is_not_stored_raises(self, storage):
        storage.add(S, context("a"))
        with pytest.raises(ValueError, match="is not stored"):
            storage.remove(S, "no-such-id")

    def test_entries_returns_a_copy(self, storage):
        storage.add(S, context("a"))
        storage.entries(S).clear()
        assert len(storage.entries(S)) == 1

    def test_clear_removes_everything(self, storage):
        for text in "abc":
            storage.add(S, context(text))
        storage.clear(S)
        assert storage.entries(S) == []
