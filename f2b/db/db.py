""" Mobility data analysis toolkit. 
"""

__authors__ = "Justine Dorsz"

from sqlite3 import Connection, Error, OperationalError, connect
from time import time

from pandas import DataFrame, Timestamp, to_datetime

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"


def create_connection(db_file: str) -> Connection:
    """
    Create a database connection to the SQLite database
        specified by the db_file.

    Args:
        db_file(str): database file path

    Return:
        Connection object or None
    """
    conn = None
    try:
        conn = connect(db_file)
    except Error as e:
        print(e)

    return conn


def get_trips_filtered_by(
    date: str,
    access_station: str = None,
    egress_station: str = None,
    time_slot_begin: str = None,
    time_slot_end: str = None,
    db_path: str = DB_PATH,
) -> DataFrame:
    """Select trips filtered by several criteria from trips table in database.
    Exclude too long trips (>3h), too short trips (< 2min).

    Args:
        - date(str): date of the trips, required argument
        - access_station(str): access station code, all stations by default
        - egress_station(str): egress station code, all stations by default
        - time_slot_begin(str): earliest access validation time,
        "00:00:00" by default
        - time_slot_end(str): latest access validation time, "23:59:59"
        by default
        - db_path(str): database location path, global DB_PATH by default

    Return:
        DataFrame: all selected trips by trip_id, with several information:
            - access_time(Timestamp): access validation time and date
            - access_station(str): access station code
            - egress_time(Timestamp): egress validation time and date
            - egress_station(str): egress station code
    """
    conn = create_connection(db_path)

    cur = conn.cursor()

    columns_of_interest = [
        "trip_id",
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
        time_slot_end_condition = 'AND "access_time" <= "' + time_slot_end + '" '

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

    trips_selected["access_time"] = date + " " + trips_selected["access_time"]
    trips_selected["access_time"] = to_datetime(trips_selected["access_time"])
    trips_selected["egress_time"] = date + " " + trips_selected["egress_time"]
    trips_selected["egress_time"] = to_datetime(trips_selected["egress_time"])

    cur.close()
    conn.close()

    return trips_selected


def get_assigned_runs_for_one_trip(trip_id: int, db_path: str = DB_PATH) -> list:
    """Get runs associated to a given trip from trip_run table in database.

    Args:
        - trip_id(int): identifier of the trip of interest
        - db_path(str): database location path, global DB_PATH by default

    Return:
        list: list of run codes of runs assigned to the trip
    """

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


def get_runs_filtered_by(
    date: str,
    direction: int = None,
    served_station_departure: str = None,
    time_slot_begin: str = None,
    time_slot_end: str = None,
    db_path: str = DB_PATH,
) -> list:
    """Select runs filtered by several criteria from runs table in database.

    Args:
        - date(str): date of the runs, required argument
        - direction(int): direction of selection, 1 eastards, 2 westwards
        - served_station_departure(str): a station at which the runs must have
         a departure record, optionnal argument
        - served_station_arrival(str): a station at which the runs must have
         a departure record, optionnal argument
        - time_slot_begin(str): earliest registered event
        "00:00:00" by default
        - time_slot_end(str): latest registered event
        - db_path(str): database location path, global DB_PATH by default

    Return:
        list: the run codes of the run satisfying the filtering conditions
    """
    conn = create_connection(db_path)

    cur = conn.cursor()

    select_columns = "SELECT DISTINCT run "

    select_columns += "FROM avl "

    date_condition = 'WHERE "date" = "' + date + '" '

    direction_condition = ""
    departure_station_condition = ""
    time_slot_begin_condition = ""
    time_slot_end_condition = ""

    if direction:
        direction_condition = 'AND "direction" = "' + str(direction) + '" '

    if served_station_departure:
        departure_station_condition = (
            'AND "station" = "' + served_station_departure + '" '
        )
        departure_station_condition += 'AND "event" = "D" '

    if time_slot_begin:
        time_slot_begin_condition = 'AND "time" >= "' + time_slot_begin + '" '
    if time_slot_end:
        time_slot_end_condition = 'AND "time" <= "' + time_slot_end + '" '

    cur.execute(
        select_columns
        + date_condition
        + direction_condition
        + departure_station_condition
        + time_slot_begin_condition
        + time_slot_end_condition
    )
    runs_selected = [run for (run,) in cur.fetchall()]

    cur.close()
    conn.close()

    return runs_selected


def get_run_arrival_time_at_station(
    run_code: str,
    station: str,
    date: str,
    db_path: str = DB_PATH,
) -> dict:
    """Select the arrival time of a given run at a given station.

    Args:
        - run_code(str): code of the run.
        - stations(str): code of the station
        - date(str): date of the day of the run
        - db_path(str): path to the database (DB_PATH by default)

    Return:
        Timestamp or None: The arrival time of the run at the station, with date and
        time at Timestamp format.
    """

    conn = create_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute("CREATE INDEX date_run_station ON avl(date, run, station)")
    except OperationalError:
        pass

    cur.execute(
        "SELECT time "
        + "FROM avl "
        + 'WHERE "date" ="'
        + date
        + '" AND "run" = "'
        + run_code
        + '" AND "station" = "'
        + station
        + '" AND "event" = "A"'
    )
    try:
        (station_arrival_time,) = cur.fetchone()

        # if record after midight, change date to next day
        if station_arrival_time >= "23:59:59":
            date = str(int(date[0:2]) + 1) + date[2:]
            station_arrival_time = (
                str(int(station_arrival_time[0:2]) - 24) + station_arrival_time[2:]
            )
        complete_arrival_time_record = Timestamp(date + " " + station_arrival_time)

    # no departure record of the run at this station
    except TypeError:
        station_arrival_time = None

    cur.close()
    conn.close()
    return complete_arrival_time_record


def get_run_departure_time_at_station(
    run_code: str,
    station: str,
    date: str,
    db_path: str = DB_PATH,
) -> dict:
    """Select the departure time of a given run at a given station.

    Args:
        - run_code (str): code of the run.
        - stations(str): code of the station
        - date(str): date of the day of the run
        - db_path(str): path to the database (DB_PATH by default)

    Return:
        Timestamp or None: The departure time of the run at the station, with date and
        time at Timestamp format.
    """

    conn = create_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute("CREATE INDEX date_run_station ON avl(date, run, station)")
    except OperationalError:
        pass

    cur.execute(
        "SELECT time "
        + "FROM avl "
        + 'WHERE "date" ="'
        + date
        + '" AND "run" = "'
        + run_code
        + '" AND "station" = "'
        + station
        + '" AND "event" = "D"'
    )
    try:
        (station_departure_time,) = cur.fetchone()

        # if record after midight, change date to next day
        if station_departure_time >= "23:59:59":
            date = str(int(date[0:2]) + 1) + date[2:]
            station_departure_time = (
                str(int(station_departure_time[0:2]) - 24) + station_departure_time[2:]
            )
        complete_departure_time_record = Timestamp(date + " " + station_departure_time)

    # no departure record of the run at this station
    except TypeError:
        station_departure_time = None

    cur.close()
    conn.close()
    return complete_departure_time_record


def get_previous_run(
    date: str,
    run_code: str,
    origin_station: str,
    destination_station: str,
    db_path: str = DB_PATH,
):
    """Get the run right before a given reference run at two served stations.

    Args:
        - date(str): date of the day of the runs
        - run_code(str): code of the reference run
        - origin_station(str): code of the first station to serve, in which
        departure times are compared
        - destination_station(str):code of the first station to serve, in which
        arrival times are compared
        - db_path(str): path to the database (DB_PATH by default)

    Return:
        str: the code of the previous run
    """
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
        + run_code
        + '" AND '
        + '"station" = "'
        + origin_station
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
        + origin_station
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
        + destination_station
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
    selection = get_runs_filtered_by(
        date="04/02/2020", direction=2, served_station_departure="VIN"
    )
    print(len(selection))
    print(f"Execution time: {time() - start_time}s.")
