from flask import Flask, request,send_from_directory
from detector import predict,train
import os

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = './uploads'

@app.route('/')
def indexApp():
    return 'service is running'

@app.route('/train')
def trainApp():
    classifier = train("./train_dir", model_save_path="trained_model.clf", n_neighbors=2,delete_unfit_files=True)
    response = send_from_directory(directory="./",filename="trained_model.clf",as_attachment=True)
    print("Training complete!")
    return response

@app.route('/detect', methods=['POST', 'GET'])
def detectApp():
    if request.method == "GET":
        return str('use post method using multipart form data, "file" as param')

        # filePath = './uploads/pic1.jpg'
        # predictions = predict(filePath, model_path ='trained_model.clf')
        # names = []
        # for name, (top, right, bottom, left) in predictions:
        #     print(name)
        #     names.append(name)
        # return str(names)

    if request.method == "POST":
        if request.files:
            image = request.files["file"]
            filePath = os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename).replace('\\', '/')
            image.save(filePath)
            print('\033[94m' + filePath + '\033[0m')
            predictions = predict(filePath, model_path ='trained_model.clf')
            names = []
            for name, (top, right, bottom, left) in predictions:
                print(name)
                names.append(name)
            return str(names)


    return 'wrong input format'


if __name__ == "__main__":
    app.run(debug=True)
