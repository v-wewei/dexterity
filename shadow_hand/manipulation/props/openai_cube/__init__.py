"""The OpenAI cube."""

from typing import Sequence, Union

from dm_control.entities.props import primitive

from shadow_hand import _SRC_ROOT

_TEXTURE_PATH = _SRC_ROOT / "manipulation" / "props" / "openai_cube" / "openai_cube.png"


class OpenAICube(primitive.Primitive):
    """A cube with OpenAI letters on each face."""

    def _build(
        self,
        size: Union[float, Sequence[float]],
        name: str = "openai_cube",
    ) -> None:
        """Builds the cube.

        Args:
            size: [x_half_length, y_half_length, z_half_length]
            name: Optional name for the cube prop.
        """
        if isinstance(size, float):
            size = [size] * 3
        else:
            if len(size) != 3:
                raise ValueError("Size must be a float or a sequence of three floats.")

        super()._build(geom_type="box", size=size, name=name)

        self.mjcf_model.asset.add(
            "texture",
            name="texture_openai",
            file=str(_TEXTURE_PATH),
            gridsize="3 4",
            gridlayout=".U..LFRB.D..",
        )
        self.mjcf_model.asset.add(
            "material",
            name="material_openai",
            texture="texture_openai",
            specular="1",
            shininess=".3",
            reflectance="0.0",
            rgba="1 1 1 1",
        )

        setattr(self.mjcf_model.find("geom", "geom"), "material", "material_openai")

    @property
    def name(self) -> str:
        return self.mjcf_model.model
