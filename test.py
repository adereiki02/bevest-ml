from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()
class Student(BaseModel):
    name: str
    roll_number: int
 
students = {
    1: Student(name="Abhishek", roll_number=1),
    2: Student(name="Ishita", roll_number=2),
    3: Student(name="Vinayak", roll_number=3),
}
@app.get("/items/{roll_number}")
def query_student_by_roll_number(roll_number: int) -> Student:
    if roll_number not in students:
        raise HTTPException(status_code=404, detail=f"Student with {roll_number=} does not exist.")
    else:
        raise HTTPException(status_code=200, detail=f"Student details are as follows: {students[roll_number]}")