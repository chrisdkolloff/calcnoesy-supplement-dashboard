import sqlite3
from typing import List, Any
import ntpath
import os
from pathlib import Path
import re
import json

import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype
from itertools import combinations
from tqdm.auto import tqdm


def connectSqlite(db):
    try:
        sqliteConnection = sqlite3.connect(db)
        # print("Connected to {} SQLite database".format(ntpath.basename(db)))
        return sqliteConnection
    except sqlite3.Error as error:
        print("Failed to connect data from {} sqlite table".format(ntpath.basename(db)), error)


def json_loads_safe(input, not_array=False):
    if input != None:
        if not_array:
            return json.loads(input) # TODO: np.ndarray
        else:
            return np.array(json.loads(input))
    else:
        return None


def fetch_table(db, table_name, not_array=False):
    """
    fetches table from SQL db and returns pd DataFrame (json_loads_safe)
    :param db:          type=str;       name of SQL db
    :param table_name:  type=str;       name of table to be fetched
    :return:            pd.DataFrame;   DataFrame containing the table (with field names)
    """
    connection = connectSqlite(db)

    table = pd.read_sql(f"SELECT * FROM {table_name}", connection)
    field_names = table.columns
    table.transpose()

    for i, field in enumerate(field_names):
        if is_string_dtype(table[field]):
            try:
                table[field] = list(map(lambda i: json_loads_safe(i, not_array), table[field].values))
            except json.decoder.JSONDecodeError:
                pass

    connection.close()
    return table


def fetch_table_where2(db, table_name, condition1, value_condition1, condition2, value_condition2):
    connection = connectSqlite(db)

    query = f"SELECT * FROM {table_name} WHERE {condition1}={value_condition1} AND {condition2}={value_condition2}"
    table = pd.read_sql(query, connection)
    field_names = table.columns
    table.transpose()

    for i, field in enumerate(field_names):
        if is_string_dtype(table[field]):
            try:
                table[field] = list(map(lambda i: json_loads_safe(i), table[field].values))
            except json.decoder.JSONDecodeError:
                pass

    connection.close()
    return table


def fetch_table_names(db):
    connection = connectSqlite(db)
    cursor = connection.cursor()

    cursor.execute(f"""SELECT name FROM sqlite_master WHERE type='table';""")
    selection = cursor.fetchall()
    selection_f = [sele[0] for sele in selection]

    cursor.close()
    connection.close()

    return selection_f


def fetch_column_names(db, table_name):
    connection = connectSqlite(db)
    cursor = connection.cursor()

    cursor.execute(f"""SELECT * FROM {table_name}""")

    cursor.close()
    connection.close()

    return [description[0] for description in cursor.description]


def fetch_column(db, table_name, column_name):
    connection = connectSqlite(db)

    query = f"SELECT {column_name} FROM {table_name}"
    table = pd.read_sql(query, connection)
    field_names = table.columns
    table.transpose()

    for i, field in enumerate(field_names):
        if is_string_dtype(table[field]):
            try:
                table[field] = list(map(lambda i: json_loads_safe(i), table[field].values))
            except json.decoder.JSONDecodeError:
                pass

    connection.close()
    return table


def is_table(db, table_name):
    connection = connectSqlite(db)
    try:
        cursor = connection.cursor()

        cursor.execute(f"""SELECT * FROM {table_name}""")

        cursor.close()
        connection.close()
        return True
    except:
        return False