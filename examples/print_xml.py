from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


def main() -> None:
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    print(hand.mjcf_model.to_xml_string())


if __name__ == "__main__":
    main()
