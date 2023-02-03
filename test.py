import datetime

now = datetime.datetime.now()
exp = datetime.datetime.strptime('2023-01-30 10:48:00', "%Y-%m-%d %H:%M:%S")
est = exp - now
print(exp, now)
print(est.total_seconds())