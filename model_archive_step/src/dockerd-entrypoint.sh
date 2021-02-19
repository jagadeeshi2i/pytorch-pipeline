#!/bin/bash
########)#############################################################
#### Script for model archiving and config.properties generation #####
######################################################################
set -e

while [[ "$#" > 0 ]]; do case $1 in
    -o|--output_path) OUTPUT_PATH="$2"; shift;shift;;
    -p|--properties) PROPERTIES="$2"; shift;shift;;
    -e|--extrafiles) EXTRAFILES="$2"; shift;shift;;
    -m|--modelfile) MODELFILE="$2"; shift;shift;;
    -s|--serializedfile) SERIALIZEDFILE="$2"; shift;shift;;
    -h|--handlerfile) HANDLERFILE="$2"; shift;shift;;
    -r|--requirements) REQUIREMENTS="$2"; shift;shift;;
  *) echo "Unknown parameter passed: $1"; shift; shift;;
esac; done

EXPORT_PATH=$OUTPUT_PATH
MODEL_STORE=$EXPORT_PATH/model-store
CONFIG_PATH=$EXPORT_PATH/config

mkdir -p $CONFIG_PATH
mkdir -p $MODEL_STORE
touch $CONFIG_PATH/config.properties


cat <<EOF > "$CONFIG_PATH"/config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
number_of_netty_threads=4
job_queue_size=100
model_store="$MODEL_STORE"
model_snapshot=
EOF

CONFIG_PROPERTIES="${CONFIG_PATH}/config.properties"
truncate -s -1 CONFIG_PROPERTIES

PROPERTIES_JSON="${PROPERTIES}/properties.json"

echo ">>>>>>>>>>>>>>>>>${PROPERTIES_JSON}"
echo ">>>>>>>>>>>>>>>>>${SERIALIZEDFILE}"

count=$(jq -c '. | length' "$PROPERTIES_JSON")
echo "{\"name\":\"startup.cfg\",\"modelCount\":\"3\",\"models\":{}}" | jq -c --arg count "${count}" '.["modelCount"]=$count' >> $CONFIG_PROPERTIES
sed -i 's/{}}//' "$CONFIG_PROPERTIES"
truncate -s -1 "$CONFIG_PROPERTIES"
echo ">>>>>>>>>Jq>>>>>>>>"
# shellcheck disable=SC1091
jq -c '.[]' "$PROPERTIES_JSON" | while read -r i; do
    modelName=$(echo "$i" | jq -r '."model-name"')
    modelFile=$(echo "$i" | jq -r '."model-file"')
    version=$(echo "$i" | jq -r '."version"')
    serializedFile=$(echo "$i" | jq -r '."serialized-file"')
    extraFiles=$(echo "$i" | jq -r '."extra-files"')
    handler=$(echo "$i" | jq -r '."handler"')
    minWorkers=$(echo "$i" | jq -r '."min-workers"')
    maxWorkers=$(echo "$i" | jq -r '."max-workers"')
    batchSize=$(echo "$i" | jq -r '."batch-size"')
    maxBatchDelay=$(echo "$i" | jq -r '."max-batch-delay"')
    responseTimeout=$(echo "$i" | jq -r '."response-timeout"')
    marName=${modelName}.mar
    requirements=$(echo "$i" | jq -r '."requirements"')
    updatedExtraFiles=$(echo "$extraFiles" | tr "," "\n" | awk -v base_path="$EXTRAFILES" '{ print base_path"/"$1 }' | paste -sd "," -)
    ########)#############################
    #### Support for custom handlers #####
    ######################################
    echo ">>>>>>>>>Hander>>>>>>>>"
    pyfile="$( cut -d '.' -f 2 <<< "$handler" )"
    if [ "$pyfile" == "py" ];
    then
        handler="${HANDLERFILE}/${handler}"
    fi
    ## 
    echo ">>>>>>>>>Requirements>>>>>>>>"
    if [ -z "${requirements}" ];    # If requirements is empty string or unset
    then
        REQUIREMENTS=$EXPORT_PATH
        touch "${REQUIREMENTS}/requirements.txt"
    fi
    echo ">>>>>>>>>MAR>>>>>>>>"
    if [ -n "${modelFile}" ];   # If modelFile is non empty string
    then
        if [ -n "${extraFiles}" ];  # If extraFiles is non empty string
        then
            torch-model-archiver --model-name "$modelName" --version "$version" --model-file "$MODELFILE/$modelFile" --serialized-file "$SERIALIZEDFILE/$serializedFile" --export-path "$MODEL_STORE" --extra-files "$updatedExtraFiles" --handler "$handler" -r "${REQUIREMENTS}/requirements.txt" --force
        else
            torch-model-archiver --model-name "$modelName" --version "$version" --model-file "$MODELFILE/$modelFile" --serialized-file "$SERIALIZEDFILE/$serializedFile" --export-path "$MODEL_STORE" --handler "$handler" -r "${REQUIREMENTS}/requirements.txt" --force
        fi
    else
        echo Model file not present
        if [ -n "${extraFiles}" ];  # If extraFiles is non empty string
        then
            torch-model-archiver --model-name "$modelName" --version "$version" --serialized-file "$SERIALIZEDFILE/$serializedFile" --export-path "$MODEL_STORE" --extra-files "$updatedExtraFiles" --handler "$handler" -r "${REQUIREMENTS}/requirements.txt" --force
        else
            torch-model-archiver --model-name "$modelName" --version "$version" --serialized-file "$SERIALIZEDFILE/$serializedFile" --export-path "$MODEL_STORE" --handler "$handler" -r "${REQUIREMENTS}/requirements.txt" --force
        fi
    fi
    echo "{\"modelName\":{\"version\":{\"defaultVersion\":true,\"marName\":\"sample.mar\",\"minWorkers\":\"sampleminWorkers\",\"maxWorkers\":\"samplemaxWorkers\",\"batchSize\":\"samplebatchSize\",\"maxBatchDelay\":\"samplemaxBatchDelay\",\"responseTimeout\":\"sampleresponseTimeout\"}}}" | 
    jq -c --arg modelName "$modelName" --arg version "$version" --arg marName "$marName" --arg minWorkers "$minWorkers" --arg maxWorkers "$maxWorkers" --arg batchSize "$batchSize" --arg maxBatchDelay "$maxBatchDelay" --arg responseTimeout "$responseTimeout" '.[$modelName]=."modelName" | .[$modelName][$version]=.[$modelName]."version" | .[$modelName][$version]."marName"=$marName | .[$modelName][$version]."minWorkers"=$minWorkers | .[$modelName][$version]."maxWorkers"=$maxWorkers | .[$modelName][$version]."batchSize"=$batchSize | .[$modelName][$version]."maxBatchDelay"=$maxBatchDelay | .[$modelName][$version]."responseTimeout"=$responseTimeout | del(."modelName", .[$modelName]."version")'  >> $CONFIG_PROPERTIES
    truncate -s -1 "$CONFIG_PROPERTIES"
done
sed -i 's/}{/,/g' "$CONFIG_PROPERTIES"
sed -i 's/}}}/}}}}/g' "$CONFIG_PROPERTIES"

# prevent docker exit
# tail -f /dev/null


