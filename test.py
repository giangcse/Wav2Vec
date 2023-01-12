# Python 3 code to demonstrate
# SHA hash algorithms.

import hashlib

# initializing string
password = "GeeksforGeeks"

# encoding GeeksforGeeks using encode()
# then sending to SHA512()

# printing the equivalent hexadecimal value.
print("The hexadecimal equivalent of SHA512 is : ")
print(hashlib.sha512(password.encode()).hexdigest())