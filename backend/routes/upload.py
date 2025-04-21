from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Načtení souboru do Pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Získání základních informací o datech
        info = {
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
        return {"filename": file.filename, "data": info}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
