from pathlib import Path
import json
_prefix=Path(__file__).with_name('run_probe.py').read_text().split("report={'scope'")[0]
exec(compile(_prefix,str(Path(__file__).with_name('run_probe.py')),'exec'))
report={'source_sha256':{'reader':copy_hash,'native_c':digest(HERE/'interpolate.c'),'library':digest(HERE/'interpolate.so')},'columns':[],'rejections':[]}
with Reader(OPTIONS['store']['path']) as table:
    for L in table.coverage_depths:
        columns,data=table._load_level(L)
        anchors=tuple(columns)
        counts=sorted({0,3,256,min(anchors,key=lambda r:abs(r-1000)),anchors[-1]})
        for r in counts:
            col=columns[r]
            nodes=col.u_min+table.grid_step*np.array([0,1,2,3,4,col.length//2,col.length-8,col.length-4,col.length-1],dtype=np.float64)
            us=np.unique(np.r_[nodes,np.nextafter(nodes,-np.inf),np.nextafter(nodes,np.inf),col.u_min-1.,80.])
            us=us[us<=table.maximum_u]
            py=original_anchor(table,L,r,us,columns,data)
            native=native_anchor(table,L,r,us,columns,data)
            report['columns'].append({'L':L,'r':r,'points':len(us),'bit_identical':py.tobytes()==native.tobytes(),'maximum_absolute_difference_nats':float(np.abs(py-native).max(initial=0.))})
        for tag,u in [('above_maximum',np.nextafter(table.maximum_u,np.inf))]:
            statuses=[]
            for implementation in (original_anchor,native_anchor):
                try:implementation(table,L,0,np.array([u]),columns,data)
                except reader.SealedTableError:statuses.append('rejected')
                else:statuses.append('accepted')
            report['rejections'].append({'L':L,'kind':tag,'outcomes':statuses})
report['all_bit_identical']=all(x['bit_identical'] for x in report['columns'])
report['all_invalid_rejected']=all(x['outcomes']==['rejected','rejected'] for x in report['rejections'])
(HERE/'boundary-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'points':sum(x['points'] for x in report['columns']),'columns':len(report['columns']),'all_bit_identical':report['all_bit_identical'],'all_invalid_rejected':report['all_invalid_rejected']}))
