import app

# This script should only be executed for debug purposes.
# The application should be launched used a production server like gunicorn.
app.application.run(debug=True, port=5000)
