In our experiments, we used version `v1.13.21` of SMDA. This version only supports PE files that require less than 100MB in memory.
The Chrome version in our dataset requires more than 100 MB in memory.
Therefore, we added a patch to SMDA that provides support for up to 100 GB in memory.

We changed the check in [line 57](https://github.com/danielplohmann/smda/blob/21cc2e538d562a8df798a64032d00929adced1c5/smda/utility/PeFileLoader.py#L57) of the [`PeFileLoader`](https://github.com/danielplohmann/smda/blob/21cc2e538d562a8df798a64032d00929adced1c5/smda/utility/PeFileLoader.py) as follows:

```python
if max_virt_section_offset and max_virt_section_offset < 100 * 1024 * 1024 * 1024:
```