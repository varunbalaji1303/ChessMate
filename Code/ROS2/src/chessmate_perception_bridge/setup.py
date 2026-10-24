from glob import glob

from setuptools import find_packages, setup


package_name = "chessmate_perception_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pesy",
    maintainer_email="pesy@example.com",
    description="File bridge from macOS ChessMate perception into ROS 2.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "move_event_bridge = chessmate_perception_bridge.move_event_bridge:main",
            "move_event_listener = chessmate_perception_bridge.move_event_listener:main",
            "verification_service = chessmate_perception_bridge.verification_service:main",
        ],
    },
)
