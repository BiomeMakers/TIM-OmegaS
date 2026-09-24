"""Lee un tensor guardado con torch.save sin necesitar torch."""
import zipfile, pickle, numpy as np, io
DT={'FloatStorage':np.float32,'DoubleStorage':np.float64,'HalfStorage':np.float16,
    'BFloat16Storage':None,'LongStorage':np.int64,'IntStorage':np.int32}
def cargar(ruta):
    z=zipfile.ZipFile(ruta); nombres=z.namelist()
    raiz=nombres[0].split('/')[0]
    pkl=[n for n in nombres if n.endswith('data.pkl')][0]
    class Almacen:
        def __init__(s,tipo,clave): s.tipo=tipo; s.clave=clave
    def rebuild(storage, offset, size, stride, *a):
        raw=z.read(f"{raiz}/data/{storage.clave}")
        if storage.tipo=='BFloat16Storage':
            u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16
            arr=u.view(np.float32)
        else:
            arr=np.frombuffer(raw,dtype=DT[storage.tipo])
        n=int(np.prod(size)) if size else 1
        return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                strides=tuple(s*arr.itemsize for s in stride)).copy() if size else arr[offset]
    class U(pickle.Unpickler):
        def find_class(s,mod,name):
            if name=='_rebuild_tensor_v2': return rebuild
            if name.endswith('Storage'): return name
            if mod=='collections' and name=='OrderedDict':
                import collections; return collections.OrderedDict
            return super().find_class(mod,name)
        def persistent_load(s,pid):
            _,tipo,clave,loc,n=pid
            if not isinstance(tipo,str): tipo=getattr(tipo,'__name__',str(tipo))
            return Almacen(tipo,clave)
    return U(io.BytesIO(z.read(pkl))).load()
