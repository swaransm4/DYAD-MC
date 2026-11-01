print('Welcome to the Python College Tracking Calculator!')
print('This program will help you choose what schools you should go to for college based on your background and major.')

user_score = 0

user_name = input('\nPlease enter your name: ')
print(f'\nHello, {user_name}! Let\'s get started.')

user_major = input('\nWhat is your intended major? ')
user_gpa = float(input('What is your current GPA?  '))
user_tests = input('Did you take the SAT or ACT? (Please type "SAT" or "ACT") ')

if user_tests.upper() == 'SAT':
    user_score = int(input('\nWhat was your SAT score? '))/1600 * 100
elif user_tests.upper() == 'ACT':
    user_score = int(input('\nWhat was your ACT score? '))/36*100

