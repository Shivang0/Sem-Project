import os
from app import app




if __name__ == '__main__':

    port = int(os.environ.get("PORT", 7000))
    app.run(threaded=True, port=port, debug=True, host= '0.0.0.0')
    #app.run(threaded=True, port=port, debug=True)