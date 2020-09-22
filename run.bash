#! /bin/bash

# ===========================================================
# Zebra Mipsology installation helper
#
# Copyright(C) 2015-2019 Mipsology SAS.  All Rights Reserved.
# ===========================================================
usage() {
  cat <<EOF

######################################################################
#   $0 - build and launch a zebra image using docker
######################################################################

Usage:

$0 [docker_run_argument] -- [commands]
  -h,--help     this help
  --runOnly     skip steps and only run the image
  --fromImg     use a tgz file to build the Zebra image
  --noZebra     do not check Zebra image on FPGA
  --dockerfile  specify another docker file to build and run
Where:
    docker_run_argument are the argument to pass to docker run command (like -v /path:/mount)
    commands            are the commands to run instead of having an interactive shell

For instance:

$0 -v /home/my_user/DATA/my_image_path:/IMAGE                   # to mount the local my_imagepath directory in /IMAGE
$0 -- '. zebra/settings.sh tensorflow ; zebra_tools --config'   # to run a command within the docker
$0 --runOnly --name httpd httpd                                 # to run an httpd docker
EOF
}

docker_run_arg=()
docker_run_cmd='/bin/bash'
runOnly=false
fromImg=false
noZebra=false
DOCKERFILE=zebra.dockerfile

while [ $# -gt 0 ]
do
    if [ "$1" = '--' ]
    then
        shift
        docker_run_cmd=''
        break
    elif [ "$1" = "--help" -o "$1" = "-h" ]
    then
      usage
      exit 0
    elif [ "$1" = "--runOnly" ]
    then
      shift
      runOnly=true
      docker_run_cmd=''
      break
    elif [ "$1" = "--fromImg" ]
    then
      shift
      fromImg=true
      continue
    elif [ "$1" = "--noZebra" ]
    then
      shift
      noZebra=true
      continue
    elif [ "$1" = "--dockerfile" ]
    then
      shift
      DOCKERFILE=$1
      shift
      continue
    fi
    docker_run_arg+=("$1")
    shift
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
ERROR="${RED}ERROR${NC}"
OK="${GREEN}OK${NC}"

APT=false
which apt &> /dev/null && APT=true

echo_error() {
  echo
  echo "============================================================"
  echo
  echo -e "$ERROR: $@"
}
echo_ok() {
  echo -e "$OK: $@"
}

fpga_check() {
  # TODO: do we want to list all the supported devices
  # we should.. but we should take the list from the kernel driver
  # lspci may be placed in /usr/sbin
  # and modinfo may be placed in /sbin
  PATH=$PATH:/sbin:/usr/sbin
  if lspci -n -d 10ee: | grep '10ee:8' &> /dev/null
  then
    echo_ok "Xilinx board detected"
  else
    echo_error "No Xilinx board detected"
    exit 1
  fi

  ABI="$( modinfo -F zebra_ABI zebra )"

  ABI_MAJOR_INSTALL=$( grep MPDD_ABI_MAJOR_VERSION $ZEBRA_INSTALL_DIR/drivers/mpdd_mapping.h | sed -e 's/"$//' -e 's/.*"//' )
  ABI_MINOR_INSTALL=$( grep MPDD_ABI_MINOR_VERSION $ZEBRA_INSTALL_DIR/drivers/mpdd_mapping.h | sed -e 's/"$//' -e 's/.*"//' )

  if [ "$ABI" != "$ABI_MAJOR_INSTALL.$ABI_MINOR_INSTALL" ]
  then
    make -C $ZEBRA_INSTALL_DIR/drivers clean all install || exit 1
  fi

  echo_ok "zebra kernel module properly installed"

  echo
  # maybe, zebra_tools is not compile to run with the libc of the host
  if ZEBRA_LOG_ENABLE=false zebra_tools --help &> /dev/null
  then
    echo "Testing board communication"
    # TODO: better check of zebra_tools --config
    (
      . $ZEBRA_INSTALL_DIR/settings.sh tensorflow INT8
      export ZEBRA_LOG_ENABLE=false
      zebra_tools --config
    )
    if [ "$?" != 0 ]
    then
      echo_error "unable to communicate with the FPGA board"
      echo "Please, make sure the proper zebra image is loaded"

      echo "refer to the error message above"
      echo "if the error still occurs, please, contact Mipsology for support"
      exit 1
    else
      echo_ok "communication with the board successful"
    fi
  fi
}

docker_check() {
  echo
  echo "Testing docker"
  if ! which docker &> /dev/null || ! docker build --help | grep -q -- '--target'
  then
    if which docker
    then
      echo_error "docker version too old. Please, remove any previous version."
      ! $APT && echo "sudo yum remove docker*"
      echo "Then, try installing the latest version using the command:"
    else
      echo_error "docker command not found. Please, install using the command:"
    fi
      $APT && echo "sudo apt update && sudo apt install -y docker.io"
    ! $APT && echo "curl -fsSL https://get.docker.com/ | sh && sudo systemctl start docker && sudo systemctl enable docker"
    exit 1
  fi

  if ! docker ps &> /dev/null
  then
    if ! ps aux | grep dockerd &> /dev/null
    then
      echo_error "docker not running. Please, try the following command"
      echo "sudo systemctl start docker"
      exit 1
    fi
    if [ "$( getent group docker )" = "" ]
    then
      echo_error "docker permission denied. Please, use the following command to have access to docker from your user"
      echo "sudo chmod a+rw /var/run/docker.sock"
      exit 1
    fi
    if getent group docker | grep -q "\b$USER\b"
    then
      echo_error "docker permission denied. Please, don't forget to logout completely from $HOSTNAME (terminate X session) and login again so the group update is taken into account"
      exit 1
    fi
    echo_error "docker permission denied. Please, make sure to be in the docker group. Use the following command, then logout and login again"
      $APT && echo "sudo adduser $USER docker"
    ! $APT && echo "sudo usermod -a -G docker $USER"
    exit 1
  fi
}


docker_build() {

  if $fromImg
  then
    nb_img="$( ls ../docker.image/*zebra_image*.tgz 2> /dev/null | wc -l )"
    [ "$nb_img" = "0" ] && echo "ERROR, no zebra image found in $ZEBRA_INSTALL_DIR/examples/docker.image. Please, copy a zebra image in the directory." && exit 1
    [ "$nb_img" != "1" ] && echo "ERROR, several zebra images found in $ZEBRA_INSTALL_DIR/examples/docker.image. Please, keep only one image in the directory." && exit 1
    zebra_image="$( ls ../docker.image/*zebra_image*.tgz 2> /dev/null )"
    # get version
    zebra_image_version="$( basename "$zebra_image" .tgz | sed 's/.*zebra_image.//' )"
    if docker inspect zebra_image:$zebra_image_version &> /dev/null
    then
      echo "Docker image is already loaded"
    else
      echo "Loading docker image $zebra_image:$zebra_image_version"
      if zcat $zebra_image | docker load
      then
        echo "loading completed"
      else
        echo_error "problem in loading the docker image"
        exit 1
      fi
    fi
    docker tag zebra_image:$zebra_image_version zebra_image
    echo "For info, the following zebra images exists in docker:"
    docker images zebra_image
    DOCKERFILE=zebra_from_img.dockerfile
    awk '/^FROM all_common AS all/{p=1;print "FROM zebra_image";next}p' < zebra.dockerfile > $DOCKERFILE
  fi
  dockertag=$( basename $DOCKERFILE .dockerfile )
  echo "building docker $dockertag using dockerfile $( basename $DOCKERFILE ) ($DOCKERFILE)"

  if ! docker build --tag=$dockertag:$UID --build-arg UID=$UID -f $DOCKERFILE .
  then
    echo_error "building docker image."
    echo "Please, review above commands and contact Mipsology for support"
    exit 1
  fi

  echo_ok "docker image built without error"
}


docker_run() {
  video_device=$( find /dev -name 'video*'  ! -perm -o+rw )
  [ "$video_device" != "" ] && echo "Giving permission to access USB webcam" && sudo chmod a+rw $video_device

  # by default we keep the same net host to be able to use open ssh X11 connection
  docker_arg="--net=host"
  # if the X11 is local, we mount also the X11 local sockets
  [ "${DISPLAY:0:1}" = ":" ] && docker_arg="$docker_arg -v /tmp/.X11-unix:/tmp/.X11-unix"
  [ -f $HOME/.Xauthority ] && docker_arg="$docker_arg -v $HOME/.Xauthority:/home/demo/.Xauthority:ro"

  [ -h $ZEBRA_INSTALL_DIR/examples/models ] && docker_arg="$docker_arg -v $( realpath $ZEBRA_INSTALL_DIR/examples/models ):$( readlink $ZEBRA_INSTALL_DIR/examples/models )"
  [ -h $ZEBRA_INSTALL_DIR/examples/datasets ] && docker_arg="$docker_arg -v $( realpath $ZEBRA_INSTALL_DIR/examples/datasets ):$( readlink $ZEBRA_INSTALL_DIR/examples/datasets )"
  [ -h $ZEBRA_INSTALL_DIR/bitstream ] && docker_arg="$docker_arg -v $( realpath $ZEBRA_INSTALL_DIR/bitstream ):$( readlink $ZEBRA_INSTALL_DIR/bitstream )"

  TTY=''
  tty &> /dev/null && TTY='-t'

  mkdir -p ~/.mipsology/zebra/log ~/.mipsology/zebra/reporting

  docker run --privileged --rm -i $TTY \
    --log-driver none \
    -e QT_X11_NO_MITSHM=1 \
    \
    -e DISPLAY=$DISPLAY \
    $docker_arg \
    \
    -v $ZEBRA_INSTALL_DIR:/home/demo/zebra \
    -v ~/.mipsology/zebra/log:/home/demo/.mipsology/zebra/log \
    -v ~/.mipsology/zebra/reporting:/home/demo/.mipsology/zebra/reporting \
    "${docker_run_arg[@]}" \
    -w /home/demo \
    $dockertag:$UID ${docker_run_cmd} "$@"
}

cd $( dirname ${BASH_SOURCE[0]} )

LOCAL_ZEBRA_INSTALL_DIR=$( cd $( dirname ${BASH_SOURCE[0]} ) ; cd ../../ ; pwd )
if [ "$ZEBRA_INSTALL_DIR" = "" ]
then
  ZEBRA_INSTALL_DIR="$LOCAL_ZEBRA_INSTALL_DIR"
else
  [ "$LOCAL_ZEBRA_INSTALL_DIR" != "$ZEBRA_INSTALL_DIR" ] &&
    echo_error "wrong zebra environment: $ZEBRA_INSTALL_DIR. Unless you are doing something special, please source environment $LOCAL_ZEBRA_INSTALL_DIR"
fi


if $runOnly
then
    docker_check
    docker run "$@"
else
    $noZebra || fpga_check
    docker_check
    docker_build
    docker_run "$@"
fi
