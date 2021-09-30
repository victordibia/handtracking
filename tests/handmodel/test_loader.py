from handtrack.handmodel import HandModel


def test_loader():
    handModel = HandModel("small")
    assert handModel.size == "small"