from absl import app
from absl import flags
from dm_control.mjcf import export_with_assets_as_zip

from dexterity import manipulation

flags.DEFINE_enum(
    "env_name",
    "reach.state_dense",
    manipulation.ALL_NAMES,
    "Which environment to load.",
)

FLAGS = flags.FLAGS


def main(_) -> None:
    domain_name, task_name = FLAGS.env_name.split(".")

    env = manipulation.load(domain_name=domain_name, task_name=task_name)

    export_with_assets_as_zip(
        mjcf_model=env.task.root_entity.mjcf_model,
        out_dir="./temp/mjcf/",
        model_name=f"{FLAGS.env_name}_export",
    )


if __name__ == "__main__":
    app.run(main)
