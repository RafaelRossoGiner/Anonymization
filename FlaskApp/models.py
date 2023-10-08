from flask_login import UserMixin
from app import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)  # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

    datasets = db.relationship('Dataset', backref='user')

    def __repr__(self):
        return f'<User "{self.name}">'


class Dataset(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)
    fileName = db.Column(db.String(100), unique=True, nullable=True)
    datasetName = db.Column(db.String(100))
    visibility = db.Column(db.Boolean, default=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    categoricalFieldMode = db.Column(db.Integer, default=0)
    dimensions = db.Column(db.Integer)
    entries = db.Column(db.Integer)

    temporary = db.Column(db.Boolean, default=False)

    TrainedSOMs = db.relationship('TrainedSOM', backref='dataset')

    def __repr__(self):
        return f'<Dataset "{self.datasetName[:20]}...">'


class TrainedSOM(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)
    k_value = db.Column(db.Integer)
    a_value = db.Column(db.Float)
    b_value = db.Column(db.Float)
    categoricalFieldMode = db.Column(db.Integer, default=0)
    sigma = db.Column(db.Integer)
    epochs = db.Column(db.Integer)
    currIterations = db.Column(db.Integer)
    yClass = db.Column(db.String(100))
    saveDate = db.Column(db.DateTime)
    ensureK = db.Column(db.Boolean, default=False)

    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))

    def __repr__(self):
        return f'<TrainedSOM "{self.id} ({self.saveDate})...">'

# class TaxonomyTree(db.Model):
#     id = db.Column(db.Integer, unique=True, primary_key=True)
#     fieldName = db.Column(db.String(100))
#
#     dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
#
#     def __repr__(self):
#         return f'<TaxonomyTree "{self.fieldName[:20]}...">'
