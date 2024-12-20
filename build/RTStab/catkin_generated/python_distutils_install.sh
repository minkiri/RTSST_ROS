#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/a/tensor_ws/src/RTStab"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/a/tensor_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/a/tensor_ws/install/lib/python3/dist-packages:/home/a/tensor_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/a/tensor_ws/build" \
    "/usr/bin/python3" \
    "/home/a/tensor_ws/src/RTStab/setup.py" \
    egg_info --egg-base /home/a/tensor_ws/build/RTStab \
    build --build-base "/home/a/tensor_ws/build/RTStab" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/a/tensor_ws/install" --install-scripts="/home/a/tensor_ws/install/bin"
