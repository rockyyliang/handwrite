import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

def save_train_history(train, val, figname, val_freq):
    train_line, = plt.plot(train, label='Training Loss')
    val_line, = plt.plot(val, label='Validation Loss')
    plt.xlabel(' x {} Batches'.format(val_freq))
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim((-2,6))
    plt.savefig(figname)

def date_string(note=None):
    '''
    function for naming things by date and time
    returns a string in the format of:
    month_date_year_hour_minute

    if note is given, it will be appended to
    the back of the string
    '''
    now = datetime.now()

    time_list = []
    #add 0 to value if single digit
    month_val = str(now.month)
    if len(month_val) == 1:
        month_val = '0' + month_val
    time_list.append(month_val)

    day_val = str(now.day)
    if len(day_val) == 1:
        day_val = '0' + day_val
    time_list.append(day_val)
    time_list.append(str(now.year))

    hour_val = str(now.hour)
    if len(hour_val) == 1:
        hour_val = '0' + hour_val
    time_list.append(hour_val)

    minute_val = str(now.minute)
    if len(minute_val) == 1:
        minute_val = str(0) + minute_val
    time_list.append(minute_val)

    #convert all to string
    time_list_str = list(map(str, time_list))
    timepath = ''
    for i in time_list_str[:-1]:
        timepath += i
        timepath += '_'
    timepath += time_list_str[-1]

    #add note if it exists
    if note is not None:
        timepath = timepath + '_' + str(note)

    return timepath
