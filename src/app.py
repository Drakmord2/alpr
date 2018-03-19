import datetime as dt
from pathlib import Path
from controller.alpr import Alpr


def main():
    image_count = 0
    start_time = dt.datetime.utcnow()

    base = Path('../base')
    images = list(base.glob('**/*.png'))

    app = Alpr()

    for i in images:
        file_name = i.parts[2]
        image_name = file_name.split('.png')[0]

        if image_name:
            image_count += 1
            print('\n[ Processing '+file_name+' ] ')
            app.process(image_name)

    stats(image_count, start_time)


def stats(image_count, start_time):
    end_time = dt.datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()

    print('\nStats:')
    print('- Images Processed: '+str(image_count))
    print('- Time elapsed: '+str(total_time)+' seconds\n')


if __name__ == '__main__':
    print('\n\t-- ALPR --')

    try:
        main()
    except Exception as err:
        print('- Runtime error: ', err)
