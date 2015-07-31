# coding: utf-8
import os
from os.path import join, exists
import hashlib
import requests
from flask import Flask
from flask import request, redirect, url_for, abort
from flask import render_template
from landmark import detectLandmarks


app = Flask(__name__)
app.config.from_object('config')

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Host': 'image.baidu.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36',
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    # check url first
    url = request.form.get('url', None)
    if url != '':
        md5 = hashlib.md5(url+app.config['MD5_SALT']).hexdigest()
        fpath = join(join(app.config['MEDIA_ROOT'], 'upload'), md5+'.jpg')
        r = os.system('wget %s -O "%s"'%(url, fpath))
        if r != 0: abort(403)
        return redirect(url_for('landmark', hash=md5))

    # save file first
    f = request.files['file']
    if f.filename == '': abort(403)
    md5 = hashlib.md5(f.filename + app.config['MD5_SALT']).hexdigest()
    fpath = join(join(app.config['MEDIA_ROOT'], 'upload'), md5+'.jpg')
    f.save(fpath)
    return redirect(url_for('landmark', hash=md5))

@app.route('/landmark/<hash>')
def landmark(hash):
    RES_ROOT = join(app.config['MEDIA_ROOT'], 'result')
    UPL_ROOT = join(app.config['MEDIA_ROOT'], 'upload')
    src = join(UPL_ROOT, hash+'.jpg')
    dst = join(RES_ROOT, hash+'.jpg')
    if not exists(src):
        abort(404)
    #if not exists(dst):
    #    detectLandmarks(src, dst)
    detectLandmarks(src, dst)
    return render_template('index.html', image=os.path.basename(dst))

@app.route('/media/result/<path>')
def serve_media(path):
    """
        Serve Media File, **this function is only used for Debug**
    """
    from flask import send_from_directory
    return send_from_directory(join(app.config['MEDIA_ROOT'], 'result'), path, \
                               as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
