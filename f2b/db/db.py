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
        "trip_id",
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
    select_columns += "FROM trip "

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

    reasonable_length_condition = 'AND "short_trip" = 0 AND "long_trip" = 0 '

    cur.execute(
        select_columns
        + date_condition
        + access_station_condition
        + egress_station_condition
        + time_slot_begin_condition
        + time_slot_end_condition
        + reasonable_length_condition
    )
    trips_selected = DataFrame(cur.fetchall(), columns=columns_of_interest)

    cur.close()
    conn.close()

    return trips_selected


def get_feasible_runs_for_one_trip(db_path: str, trip_id: int) -> list:
    conn = create_connection(db_path)
    cur = conn.cursor()

    try:
        cur.execute("CREATE INDEX trip_index ON trip_run(trip_id)")
    except OperationalError:
        pass

    cur.execute(f'SELECT run FROM trip_run WHERE "trip_id" = {trip_id}')
    request_result = cur.fetchall()
    runs = [run[0] for run in request_result]

    cur.close()
    conn.commit()
    conn.close()
    return runs


def get_run_arrivals(db_path: str, date: str, run: str, stations: list):
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
            (station_arrival_time,) = cur.fetchone()
        except TypeError:  # no arrival record at this station
            station_arrival_time = None

        run_info[run, station] = station_arrival_time

    cur.close()
    conn.close()
    return run_info


def get_run_departures(db_path: str, date: str, run: str, stations: list):
    conn = create_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute("CREATE INDEX date_run_station ON avl(date, run, station)")
    except OperationalError:
        pass

    run_departures = {}
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
            + '" AND "event" = "D"'
        )
        try:
            (station_departure_time,) = cur.fetchone()
        except TypeError:  # no departure record at this station
            station_departure_time = None

        run_departures[run, station] = station_departure_time

    cur.close()
    conn.close()
    return run_departures


def get_previous_run(
    db_path: str,
    date: str,
    run: str,
    station_origin: str,
    station_destination: str,
):
    conn = create_connection(db_path)
    cur = conn.cursor()

    try:
        cur.execute(
            "CREATE INDEX date_run_station_event_direction_time "
            + "ON avl(date, run, station, event, direction, time"
        )
    except OperationalError:
        pass

    query_departure_time_direction = (
        "SELECT time, direction FROM avl "
        + "WHERE "
        + '"date" = "'
        + date
        + '" AND '
        + '"run" = "'
        + run
        + '" AND '
        + '"station" = "'
        + station_origin
        + '" AND '
        + '"event" = "D"'
    )
    cur.execute(query_departure_time_direction)
    (departure_time, direction) = cur.fetchone()

    query_select_candidate_runs_departure = (
        "SELECT run FROM avl "
        + "WHERE "
        + '"date" = "'
        + date
        + '" AND '
        + '"run" != "      " '
        + " AND "
        + '"station" = "'
        + station_origin
        + '" AND '
        + '"event" = "D" '
        + "AND "
        + '"direction" = "'
        + direction
        + '" AND '
        + '"time" < "'
        + departure_time
        + '" '
        + "ORDER BY time DESC"
    )
    cur.execute(query_select_candidate_runs_departure)
    candidate_runs_departure = cur.fetchall()

    query_select_candidate_runs_arrival = (
        "SELECT run FROM avl "
        + "WHERE "
        + '"date" = "'
        + date
        + '" AND '
        + '"run" != "      " '
        + " AND "
        + '"station" = "'
        + station_destination
        + '" AND '
        + '"event" = "A" '
        + "AND "
        + '"direction" = "'
        + direction
        + '"'
    )

    cur.execute(query_select_candidate_runs_arrival)
    candidate_runs_arrival = cur.fetchall()

    previous_run = None
    for (run,) in candidate_runs_departure:
        if (run,) in candidate_runs_arrival:
            previous_run = run
            break

    cur.close()
    conn.close()
    return previous_run


if __name__ == "__main__":
    start_time = time()
    db_path = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"
    # selection = get_trips_filtered_by(db_path, time_slot_begin="13:05:20")
    selection = get_previous_run(db_path, "01/02/2020", "UBOS39", "VIN", "NAT")
    print(selection)
    print(f"Execution time: {time() - start_time}s.")
