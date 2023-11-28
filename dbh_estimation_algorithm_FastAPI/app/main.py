from fastapi import FastAPI, File, UploadFile
from app.scripts import helpers, segmentation
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

@app.get("/")
def read_root():
    return {"response": "Server is running"}


@app.post("/recognize/{measured_dbh}")
async def recognize(measured_dbh: float ,uploaded_file: UploadFile = File(...)):
    
    # generate a temporal filename
    #filename = f"{helpers.generate_random_file_name()}@{uploaded_file.filename}"
    filename = uploaded_file.filename
    file_location = f"app/data/{filename}"
    
    # save file temporally
    helpers.saveUploadfile(file_location, uploaded_file)

    # run model on saved image
    dbh = segmentation.getTreeDBH2(file_location, measured_dbh)

    #print(measured_dbh)

    helpers.removefile(file_location)
    # upload file to s3 and delete from local drive in the background
    #background_task.add_task(uploadfileToS3, file_location, filename, text)

    return {
            "dbh": dbh
        }