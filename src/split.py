"""Grouped train/validation splitting.

Rules from the spec:
- annotation sets of the same observation must never be split across train and val;
- observations close in time, or from the same instrument, are not independent,
  so group by time window and/or instrument as well.
"""
