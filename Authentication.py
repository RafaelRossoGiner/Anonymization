import os
import shutil

from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
import models
import Config as C
from app import db

Auth = Blueprint('Auth', __name__)


@Auth.route('/login')
def login():
    return render_template('login.html')


@Auth.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = models.User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('Auth.login'))  # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    # create neccesary directories
    os.makedirs(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(user.id)), 0o777, True)
    login_user(user, remember=remember)
    return redirect(url_for('FrontEndWeb.profile'))


@Auth.route('/signup')
def signup():
    return render_template('signup.html')


@Auth.route('/signup', methods=['POST'])
def signup_post():
    # code to validate and add user to database goes here
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    confirmPassword = request.form.get('confirmPassword')

    if password != confirmPassword:
        flash('Passwords do not match')
        return redirect(url_for('Auth.signup'))

    user = models.User.query.filter((models.User.email == email) | (models.User.name == name)).first()

    if user:  # if a user is found, we want to redirect back to signup page so user can try again
        if user.email == email:
            flash('Email address already exists')
        else:
            flash('Name already exists')
        return redirect(url_for('Auth.signup'))

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    new_user = models.User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    # create neccesary directories
    os.makedirs(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(new_user.id)), 0o777, True)

    return redirect(url_for('Auth.login'))


@Auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('Auth.login'))


@Auth.route('/profileUpdate', methods=['POST'])
@login_required
def modify_account():
    if request.form.get('updateName') is not None:
        field = 'name'
        newValue = request.form.get('newName')
        user = models.User.query.filter((models.User.name == newValue)).first()
        if user:
            flash('Name already exists', 'warning')
            return redirect(request.referrer)
    elif request.form.get('updateEmail') is not None:
        field = 'email'
        newValue = request.form.get('newEmail')
        user = models.User.query.filter((models.User.email == newValue)).first()
        if user:
            flash('Email address already exists', 'warning')
            return redirect(request.referrer)
    elif request.form.get('updatePassword') is not None:
        field = 'password'
        password = request.form.get('newPassword')
        passwordConfirmation = request.form.get('confirmPassword')
        if password != passwordConfirmation:
            flash('Passwords do not match', 'warning')
            return redirect(request.referrer)
        newValue = generate_password_hash(password, method='sha256')
    elif request.form.get('deleteAccount') is not None:
        if request.form.get('ConfirmDelete') is None:
            flash('You must check the confirmation box to succesfully delete your account', 'success')
            return redirect(request.referrer)

        user_id = current_user.id
        logout_user()

        User = db.session.query(models.User).get(user_id)
        db.session.delete(User)
        db.session.commit()

        # Delete user directories
        imageDir = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(user_id))
        shutil.rmtree(imageDir)

        flash('Account succesfully deleted')
        return redirect(url_for('Auth.login'))

    User = db.session.query(models.User).get(current_user.id)
    User.__setattr__(field, newValue)
    db.session.commit()

    flash('Account updated!', 'success')
    return redirect(url_for('FrontEndWeb.profile'))
