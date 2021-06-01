if [ -z "$1" ]
  then
    echo "Port not specified"
    exit 11
fi

PROJECTPORT=$1

setsid nohup kubectl port-forward service/antyrec $PROJECTPORT:8006 > nohup.out &
