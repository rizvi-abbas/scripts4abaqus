# scripts4abaqus

```python
writeFieldTensors1('Job-6.odb',f"abc.zip",['E_maxPrincipal','E_minPrincipal'])
```

```python
create_odb_from_inp1("abc.inp", "Job-new.odb")
add_result_to_odb1('Job-new.odb', "abc.zip", fields=['E_maxPrincipal','E_minPrincipal'])
```
