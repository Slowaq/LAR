#!/bin/bash
# file: install.sh
# desc: Turtlebot python wrapper install script
# auth: Libor Wagner <libor.wagner@cvut.cz>

#%% config

BASEDIR=$HOME/turtle_test
REPO=git@gitlab.fel.cvut.cz:wagnelib/turtlebot.git
ROSVER=melodic
ROSBASE=/opt/ros/$ROSVER
ROSWS=$BASEDIR/lar_ws

#%% functions
function log_err()
{
    echo -e "\e[31m$1\e[0m"
}

function log_warn()
{
    echo -e "\e[93m$1\e[0m"
}

#%% pre-check

CHECK=true

# check ros installed
if [ ! -d "$ROSBASE" ]; then
    log_err "ROS not installed, $ROSBASE does not exist\!"
    CHECK=false
fi 

# TODO: check kobuki package installed

# check whether the ws exist
if [[ -d "$ROSWS" ]]; then
    log_err "ROS workspace $ROSWS already exists\!"
    CHECK=false
fi

# final check
if [[ $CHECK==false ]]; then
    log_err "Failed"
    exit 1
fi

#%% ros ws

# source ros environment
source $ROSBASE/setup.bash || log_warn "source $ROSBASE/setup.bash failed"

# make workspace directory structure
mkdir -p $ROSWS/src || log_warn "mkdir -p $ROSWS/src failed"

# close turtlebot repo
git clone $REPO $ROSWS/src/robolab_turtlebot || log_warn "git clone $REPO $ROSWS/src/robolab_turtlebot failed"

# make build the workspace
(cd $ROSWS
    catkin_init_workspace || log_warn "catkin init_workspace failed"
    catkin build || log_warn "catkin build failed"
)

# add stuff to .bashrc 

echo 'source $HOME/devel/setup.bash' >> $BASEDIR/.bashrc
echo 'export PATH=$PATH:'$ROSWS'/src/robolab_turtlebot/bin' >> $BASEDIR/.bashrc

#$$ post-check
