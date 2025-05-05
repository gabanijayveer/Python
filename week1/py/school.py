class Student:
    def __init__(self, name):
        self.name = name
        self.courses = []
    def enroll(self, course):
        self.courses.append(course)
        print(f"{self.name} has enrolled in {course.name}.")
    def get_courses(self):
        return [course.name for course in self.courses]
class Faculty:
    def __init__(self, name):
        self.name = name
class Course:
    def __init__(self, name):
        self.name = name
        self.faculty = None  # Associate a faculty member with the course
    def assign_faculty(self, faculty):
        self.faculty = faculty
        print(f"Faculty {faculty.name} assigned to course {self.name}.")
class School:
    def __init__(self):
        self.students = {}
        self.courses = {}
        self.faculties = {}
    def add_faculty(self, f_name):
        if f_name not in self.faculties:
            self.faculties[f_name] = Faculty(f_name)
            print(f"Faculty {f_name} added.")
        else:
            print("Faculty already exists.")
    def add_student(self, name):
        if name not in self.students:
            self.students[name] = Student(name)
            print(f"Student {name} added.")
        else:
            print("Student already exists.")
    def add_course(self, name, f_name):
        if name not in self.courses:
            course = Course(name)
            self.courses[name] = course
            if f_name in self.faculties:
                course.assign_faculty(self.faculties[f_name])
            else:
                print(f"Faculty {f_name} does not exist. Course created without faculty assignment.")
            print(f"Course {name} added.")
        else:
            print("Course already exists.")
    def enroll_student_in_course(self, student_name, course_name):
        if student_name in self.students and course_name in self.courses:
            self.students[student_name].enroll(self.courses[course_name])
        else:
            print("Student or Course not found.")
# Main loop
school = School()
while True:
    print("\n1. Add Student")
    print("2. Add Course")
    print("3. Enroll Student in Course")
    print("4. Show Student Courses")
    print("5. Add Faculty")
    print("6. Exit")
    
    choice = input("Choose an option: ")
    if choice == "1":
        student_name = input("Enter student name: ")
        school.add_student(student_name)
    elif choice == "2":
        course_name = input("Enter course name: ")
        faculty_name = input("Enter faculty name: ")
        school.add_course(course_name, faculty_name)
    elif choice == "3":
        student_name = input("Enter student name: ")
        course_name = input("Enter course name: ")
        school.enroll_student_in_course(student_name, course_name)
    elif choice == "4":
        student_name = input("Enter student name: ")
        if student_name in school.students:
            courses = school.students[student_name].get_courses()
            print(f"{student_name} is enrolled in: {', '.join(courses)}")
        else:
            print("Student not found.")
    elif choice == "5":
        faculty_name = input("Enter faculty name: ")
        school.add_faculty(faculty_name)
    elif choice == "6":
        print("Exiting the school management system.")
        break
    else:
        print("Invalid option. Please try again.")
