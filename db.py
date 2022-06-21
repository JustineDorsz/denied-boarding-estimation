""" Mobility data analysis toolkit. 
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-06-02"

from sqlite3 import Connection, Error, connect, OperationalError
from time import time

from pandas import DataFrame

# --------------------------------------------------------------------------------
# Issue:
# tradeoff between query genericity and number of queries (performance)
#
# --------------------------------------------------------------------------------

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"


def create_connection(db_file: str) -> Connection:
    """
    Create a database connection to the SQLite database
        specified by the db_file.
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = connect(db_file)
    except Error as e:
        print(e)

    return conn


def get_trips_filtered_by(
    db_path: str,
    date: str,
    access_station: str = None,
    egress_station: str = None,
    time_slot_begin: str = None,
    time_slot_end: str = None,
    feasible_runs_nbr: int = None,
) -> DataFrame:
    conn = create_connection(db_path)

    cur = conn.cursor()

    columns_of_interest = [
        "id",
        "navigo",
        "date",
        "access_time",
        "access_station",
        "egress_time",
        "egress_station",
    ]

    select_columns = "SELECT "
    for field in columns_of_interest:
        select_columns += field
        if field != columns_of_interest[-1]:
            select_columns += ", "
        else:
            select_columns += " "
    select_columns += "FROM trips "

    date_condition = 'WHERE "date" = "' + date + '" '

    access_station_condition = ""
    egress_station_condition = ""
    time_slot_begin_condition = ""
    time_slot_end_condition = ""

    if access_station:
        access_station_condition = 'AND "access_station" = "' + access_station + '" '
    if egress_station:
        egress_station_condition = 'AND "egress_station" = "' + egress_station + '" '
    if time_slot_begin:
        time_slot_begin_condition = 'AND "access_time" >= "' + time_slot_begin + '" '
    if time_slot_end:
        time_slot_end_condition = 'AND "egress_time" <= "' + time_slot_end + '" '

    cur.execute(
        select_columns
        + date_condition
        + access_station_condition
        + egress_station_condition
        + time_slot_begin_condition
        + time_slot_end_condition
    )
    trips_selected = DataFrame(cur.fetchall(), columns=columns_of_interest)

    cur.close()
    conn.close()

    return trips_selected


def get_feasible_runs_for_one_trip(db_path: str, trip_id: int) -> list:
    conn = create_connection(db_path)
    cur = conn.cursor()

    try:
        cur.execute("CREATE INDEX trip_index ON trip_run(trip_index)")
    except OperationalError:
        pass

    cur.execute(f'SELECT mission FROM trip_run WHERE "trip_index" = {trip_id}')
    request_result = cur.fetchall()
    runs = [run[0] for run in request_result]

    cur.close()
    conn.commit()
    conn.close()
    return runs


def get_run_info(db_path: str, date: str, run: str, stations: list):
    conn = create_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute("CREATE INDEX date_run_station ON avl(date, run, station)")
    except OperationalError:
        pass

    run_info = {}
    for station in stations:
        cur.execute(
            "SELECT time "
            + "FROM avl "
            + 'WHERE "date" ="'
            + date
            + '" AND "run" = "'
            + run
            + '" AND "station" = "'
            + station
            + '" AND "event" = "A"'
        )
        try:
            station_arrival = cur.fetchall()[0][0]
        except IndexError:
            station_arrival = None

        cur.execute(
            "SELECT time "
            + "FROM avl "
            + 'WHERE "date" ="'
            + date
            + '" AND "run" = "'
            + run
            + '" AND "station" = "'
            + station
            + '" AND "event" = "D"'
        )
        try:
            station_departure = cur.fetchall()[0][0]
        except IndexError:
            station_departure = None

        run_info[date, run, station] = (station_arrival, station_departure)

    return run_info


if __name__ == "__main__":
    start_time = time()
    db_path = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"
    # selection = get_trips_filtered_by(db_path, time_slot_begin="13:05:20")
    selection = get_run_info(db_path, "03/02/2020", "ZEBU01", ["NAV", "RUE"])
    print(selection)
    print(f"Execution time: {time() - start_time}s.")
