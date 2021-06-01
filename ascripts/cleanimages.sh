GITLAB_REGISTRY=$1
UNAME=$2
PROJECT=$3

docker images | grep $GITLAB_REGISTRY/$UNAME/$PROJECT | awk '{print $1 ":" $2}' | xargs docker rmi