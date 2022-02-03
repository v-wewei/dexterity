import matplotlib.pyplot as plt
from dm_control import mjcf

from shadow_hand import shadow_hand_e
from shadow_hand import shadow_hand_e_constants as consts


def render(physics: mjcf.Physics, name: str = "") -> None:
    pixels = physics.render(width=640, height=480, camera_id="cam0")
    _ = plt.figure(name)
    plt.imshow(pixels)
    plt.show()


def main() -> None:
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    physics.step()
    render(physics, "initial")


if __name__ == "__main__":
    main()
