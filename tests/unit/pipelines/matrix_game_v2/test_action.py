from telefuser.pipelines.matrix_game_v2.action import MatrixGameAction


def test_matrix_action_normalizes_browser_keys_and_mouse() -> None:
    action = MatrixGameAction.from_payload(
        {
            "action": {"keys": ["ArrowUp", "KeyD", "idle"]},
            "mouse": {"x": 1.5, "y": -2},
        }
    )

    assert action.keyboard == ("w", "d")
    assert action.mouse_dx == 1.5
    assert action.mouse_dy == -2.0
    assert action.as_conditions() == {
        "keyboard": ["w", "d"],
        "mouse": {"dx": 1.5, "dy": -2.0},
    }


def test_matrix_action_accepts_neutral_control() -> None:
    assert MatrixGameAction.from_payload({"controls": "idle"}) == MatrixGameAction()
