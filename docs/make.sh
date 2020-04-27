VIRTUALENV=/media/pop_data/.venv/uqa
DOCS_FOLDER=/media/pop_data/repos/uqa/docs
TARGET=html

source $VIRTUALENV/bin/activate
make -C $DOCS_FOLDER $TARGET