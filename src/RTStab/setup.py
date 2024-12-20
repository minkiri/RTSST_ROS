from setuptools import setup

setup(
    name='RTStab',
    version='0.0.1',
    packages=['RTStab'],
    package_dir={'': 'src'},
    install_requires=['rospy', 'cv_bridge', 'torch'],  # 필요한 패키지 추가
)
