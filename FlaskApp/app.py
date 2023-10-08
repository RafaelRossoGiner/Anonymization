import logging
import os
import shutil
import sys
import traceback

# from logging import FileHandler, WARNING
from flask import Flask
from flask_executor import Executor
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import Config as C

# Create DB
db = SQLAlchemy()

# Create process executor
executor = Executor()


def configure_app():
    # Build directory structure
    BuildDirStructure()

    # Erase any temporary files left
    EraseDir(C.DATASETS_TEMPORARY_PATH)
    EraseDir(C.DATASETS_RESULTS_PATH)

    flask_app = Flask(__name__, static_url_path='/static', instance_relative_config=True)

    flask_app.config.from_pyfile('flask_settings.py', silent=False)

    # Configure app executor for multiprocess
    executor.init_app(flask_app)
    flask_app.config['EXECUTOR_TYPE'] = 'process'
    flask_app.config['EXECUTOR_MAX_WORKERS'] = C.FLASK_WORKERS

    # Configure app database
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = C.DataBaseURI

    flask_app.config["SECRET_KEY"] = C.SecretKey

    flask_app.config['UPLOAD_FOLDER'] = C.DATASETS_TEMPORARY_PATH

    # Disable Logger
    if not C.LOGGING_ENABLED:
        logging.getLogger('werkzeug').disabled = True

    if C.LOGGING_ENABLED:
        SetupLogger(flask_app.root_path)

    db_engine = create_engine(C.DataBaseURI)
    session_factory = sessionmaker(bind=db_engine)
    db.Session = scoped_session(session_factory)

    db.init_app(flask_app)

    login_manager = LoginManager()
    login_manager.login_view = 'Auth.login'
    login_manager.init_app(flask_app)

    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    with flask_app.app_context():
        db.create_all()

    # blueprint for auth routes in our app
    from Authentication import Auth as Auth_blueprint
    flask_app.register_blueprint(Auth_blueprint)

    # blueprint for non-auth parts of app
    from FrontEndWeb import FrontEndWeb as FrontEndWeb_blueprint
    flask_app.register_blueprint(FrontEndWeb_blueprint)

    return flask_app


# log uncaught exceptions with traceback
def log_exceptions_trace(type, value, tb):
    for line in traceback.TracebackException(type, value, tb).format(chain=True):
        logging.exception(line)
    logging.exception(value)

    sys.__excepthook__(type, value, tb)  # calls default excepthook


# log uncaught exceptions without traceback
def log_exceptions(type, value, tb):
    logging.exception(value)
    sys.__excepthook__(type, value, tb)  # calls default excepthook


def SetupLogger(basePath):
    file_path = os.path.join(basePath, C.LOGGING_FILENAME)

    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

    logging.basicConfig(filename=C.LOGGING_FILENAME,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=int(C.LOGGING_LEVEL) * 10)

    logging.info("Running Urban Planning")
    if C.LOGGING_TRACEBACK:
        sys.excepthook = log_exceptions_trace
    else:
        sys.excepthook = log_exceptions


def SyncFilesWithDatabase(app_selected):
    import models

    with app_selected.app_context():
        # Remove any db item without corresponding file
        datasets = db.session.query(models.Dataset).all()
        for dataset in datasets:
            datasetPath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
            if not os.path.isfile(datasetPath):
                db.session.delete(dataset)

        db.session.commit()

        # Remove any file without corresponding db item
        dirFiles = os.scandir(C.DATASETS_SAVE_PATH)
        for entry in dirFiles:
            if entry.is_file():
                datasetName = os.path.split(entry.path)[1]
                datasetID = datasetName.replace(C.DATASET_FILENAME_PREFIX, '')
                exists = db.session.query(models.Dataset).filter(
                    models.Dataset.id == int(datasetID)).first() is not None
                if not exists:
                    # Remove dataset
                    try:
                        if os.path.isfile(entry.path) or os.path.islink(entry.path):
                            os.unlink(entry.path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (entry.path, e))

                    try:
                        # Remove trees
                        treesDir = os.path.join(app_selected.root_path, C.TAXONOMY_TREE_PATH,
                                                C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(datasetID))
                        shutil.rmtree(treesDir)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (entry.path, e))

    dirFiles.close()


def BuildDirStructure():
    import Config as C

    mode = 0o777
    allowExisting = True

    os.makedirs(C.DATASETS_SAVE_PATH, mode, allowExisting)
    os.makedirs(C.DATASETS_TEMPORARY_PATH, mode, allowExisting)
    os.makedirs(C.DATASETS_RESULTS_PATH, mode, allowExisting)
    os.makedirs(C.DATASETS_TRAINED_POINTS_PATH, mode, allowExisting)
    os.makedirs(C.TAXONOMY_TREE_PATH, mode, allowExisting)


def EraseDir(dirPath):
    dirFiles = os.scandir(dirPath)
    for entry in dirFiles:
        if entry.is_file():
            file_path = entry.path
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    dirFiles.close()


def create_app():
    app = configure_app()

    # Delete any dataset without a corresponding file
    SyncFilesWithDatabase(app)

    return app


application = create_app()
