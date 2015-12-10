"""
Thierry Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
This code uses 7digital API and info contained in HDF5 song
file to get a preview URL.
This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.
Copyright 2010, Thierry Bertin-Mahieux
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import urllib2
from urllib2 import HTTPError
import tables
import py7D
try:
    from numpy import argmin
except ImportError:
    from scipy import argmin
except ImportError:
    print 'no argmin function (no numpy or scipy), might cause problems'
from xml.dom import minidom

global DIGITAL7_API_KEY
DIGITAL7_API_KEY = "7daju79vrfv8"

def set_credentials(key, secret):
    py7D.set_credentials(key, secret)
    DIGITAL7_API_KEY = key

def url_call(url):
    """
    Do a simple request to the 7digital API
    We assume we don't do intense querying, this function is not
    robust
    Return the answer as na xml document
    """
    try:
        stream = urllib2.urlopen(url)
    except HTTPError,e:
        print "\t",e
        print "\t%s"%url
        raise e
    xmldoc = minidom.parse(stream).documentElement
    stream.close()
    return xmldoc


def levenshtein(s1, s2):
    """
    Levenstein distance, or edit distance, taken from Wikibooks:
    http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s1:
        return len(s2)
 
    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


def get_closest_track(tracklist,target):
    """
    Find the closest track based on edit distance
    Might not be an exact match, you should check!
    """
    dists = map(lambda x: levenshtein(x,target),tracklist)
    best = argmin(dists)
    return tracklist[best]


def get_trackid_from_text_search(title,artistname=''):
    """
    Search for an artist + title using 7digital search API
    Return None if there is a problem, or tuple (title,trackid)
    """
    url = 'http://api.7digital.com/1.2/track/search?'
    url += 'oauth_consumer_key='+DIGITAL7_API_KEY
    query = title
    if artistname != '':
        query = artistname + ' ' + query
    query = urllib2.quote(query)
    url += '&q='+query
    xmldoc = url_call(url)
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
    resultelem = xmldoc.getElementsByTagName('searchResult')
    if len(resultelem) == 0:
        return None
    track = resultelem[0].getElementsByTagName('track')[0]
    tracktitle = track.getElementsByTagName('title')[0].firstChild.data
    trackid = int(track.getAttribute('id'))
    return (tracktitle,trackid)

    
def get_tracks_from_artistid(artistid):
    """
    We get a list of release from artists.
    For each of these, get release.
    After calling the API with a given release ID, we receive a list of tracks.
    We return a map of <track name> -> <track id>
    or None if there is a problem
    """
    url = 'http://api.7digital.com/1.2/artist/releases?'
    url += '&artistid='+str(artistid)
    url += '&country=GB'
    url += '&oauth_consumer_key='+DIGITAL7_API_KEY
    xmldoc = url_call(url)
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
    releaseselem = xmldoc.getElementsByTagName('releases')[0]
    releases = releaseselem.getElementsByTagName('release')
    if len(releases) == 0:
        return None
    releases_ids = map(lambda x: int(x.getAttribute('id')), releases)
    res = {}
    for rid in releases_ids:
        tmpres = get_tracks_from_releaseid(rid)
        if tmpres is not None:
            res.update(tmpres)
    return res


def get_tracks_from_releaseid(releaseid):
    """
    After calling the API with a given release ID, we receive a list of tracks.
    We return a map of <track name> -> <track id>
    or None if there is a problem
    """
    url = 'http://api.7digital.com/1.2/release/tracks?'
    url += 'releaseid='+str(releaseid)
    url += '&country=GB'
    url += '&oauth_consumer_key='+DIGITAL7_API_KEY
    xmldoc = url_call(url)
    #print xmldoc.toprettyxml()
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
    tracks = xmldoc.getElementsByTagName('track')
    if len(tracks)==0:
        return None
    res = {}
    for t in tracks:
        tracktitle = t.getElementsByTagName('title')[0].firstChild.data
        trackid = int(t.getAttribute('id'))
        res[tracktitle] = trackid
    return res
    

def get_preview_from_trackid(trackid):
    return py7D.preview_url(trackid)


def die_with_usage():
    """ HELP MENU """
    print 'get_preview_url.py'
    print '    by T. Bertin-Mahieux (2010) Columbia University'
    print 'HELP MENU'
    print 'usage:'
    print '    python get_preview_url.py [FLAG] <SONGFILE>'
    print 'PARAMS:'
    print '  <SONGFILE>  - a Million Song Dataset file TRABC...123.h5'
    print 'FLAGS:'
    print '  -7digitalkey KEY - API key from 7 digital, we recomment you put it'
    print '                     under environment variable: DIGITAL7_API_KEY'
    print 'OUTPUT:'
    print '  url from 7digital that should play a clip of the song.'
    print '  No guarantee that this is the exact audio used for the analysis'
    sys.exit(0)


def get_url(h5path):


    if not os.path.isfile(h5path):
        print 'invalid path (not a file):',h5path


    # open h5 song, get all we know about the song
    h5 = tables.open_file(h5path,mode='r') 
    track_7digitalid = h5.root.metadata.songs.cols.track_7digitalid[0]
    release_7digitalid = h5.root.metadata.songs.cols.release_7digitalid[0]
    artist_7digitalid = h5.root.metadata.songs.cols.artist_7digitalid[0]
    artist_name = h5.root.metadata.songs.cols.artist_name[0]
    release_name = h5.root.metadata.songs.cols.release[0]
    track_name = h5.root.metadata.songs.cols.title[0]
    h5.close()

    # we already have the 7digital track id? way too easy!
    if track_7digitalid >= 0 :
        preview = get_preview_from_trackid(track_7digitalid)
        if preview == '':
            print 'something went wrong when looking by track id'
        else:
            return preview

    # we have the release id? get all tracks, find the closest match
    if release_7digitalid >= 0:
        tracks_name_ids = get_tracks_from_releaseid(release_7digitalid)
        if tracks_name_ids is None:
            print 'something went wrong when looking by album id'
        else:
            closest_track = get_closest_track(tracks_name_ids.keys(),track_name)
            if closest_track != track_name:
                print 'we approximate your song title:',track_name,'by:',closest_track
            preview = get_preview_from_trackid(tracks_name_ids[closest_track])
            if preview == '':
                print 'something went wrong when looking by track id after release id'
            else:
                return preview
            
    # we have the artist id? get all albums, get all tracks, find the closest match
    if artist_7digitalid >= 0:
        tracks_name_ids = get_tracks_from_artistid(artist_7digitalid)
        if tracks_name_ids is None:
            print 'something went wrong when looking by artist id'
        else:
            closest_track = get_closest_track(tracks_name_ids.keys(),track_name)
            if closest_track != track_name:
                print 'we approximate your song title:',track_name,'by:',closest_track
            preview = get_preview_from_trackid(tracks_name_ids[closest_track])
            if preview == '':
                print 'something went wrong when looking by track id after artist id'
            else:
                return preview

    # damn it! search by artist name + track title
    else:
        res = get_trackid_from_text_search(track_name,artistname=artist_name)
        if res is None:
            print 'something went wrong when doing text search with artist and track name, no more ideas'
        closest_track,trackid = res
        if closest_track != track_name:
            print 'we approximate your song title:',track_name,'by:',closest_track
        preview = get_preview_from_trackid(trackid)
        if preview == '':
            print 'something went wrong when looking by track id after text searching by artist and track name'
        else:
            return preview

def get_url_simple(track):
    artist_name = track['artist']
    track_name = track['title']
    
    print "\tusing artist and title: %s by %s" % (track_name, artist_name)
    res = get_trackid_from_text_search(track_name,artistname=artist_name)
    if res is None:
        print '\tsomething went wrong when doing text search with artist and track name, no more ideas'
        return None, -1
    closest_track,trackid = res
    if closest_track != track_name:
        print '\twe approximate your song title:',track_name,'by:',closest_track
    preview = get_preview_from_trackid(trackid)
    if preview == '':
        print '\tsomething went wrong when looking by track id after text searching by artist and track name'
    else:
        return preview, trackid
    
    return None, -1

def get_url_from_dict(track, deprecated_id = False):
    #same as above but gets data from dict
    
    track_7digitalid = track['track_7digitalid']
    release_7digitalid = track['release_7digitalid']
    artist_7digitalid = track['artist_7digitalid']
    artist_name = track['artist']
    release_name = track['release']
    track_name = track['title']

    # we already have the 7digital track id? way too easy!
    if track_7digitalid >= 0 and not deprecated_id:
        print "\ttrack has id %d" % track_7digitalid
        preview = get_preview_from_trackid(track_7digitalid)
        if preview == '':
            print '\tsomething went wrong when looking by track id'
        else:
            return preview, track_7digitalid

    # we have the release id? get all tracks, find the closest match
    if release_7digitalid >= 0:
        print "\tusing release id: %d" % release_7digitalid
        tracks_name_ids = get_tracks_from_releaseid(release_7digitalid)
        if tracks_name_ids is None:
            print '\tsomething went wrong when looking by album id'
        else:
            closest_track = get_closest_track(tracks_name_ids.keys(),track_name)
            if closest_track != track_name:
                print '\twe approximate your song title:',track_name,'by:',closest_track
            preview = get_preview_from_trackid(tracks_name_ids[closest_track])
            if preview == '':
                print '\tsomething went wrong when looking by track id after release id'
            else:
                return preview, tracks_name_ids[closest_track]
            
    # we have the artist id? get all albums, get all tracks, find the closest match
    if artist_7digitalid >= 0:
        print "\tusing artist id: %d" % artist_7digitalid
        tracks_name_ids = get_tracks_from_artistid(artist_7digitalid)
        if tracks_name_ids is None:
            print '\tsomething went wrong when looking by artist id'
        else:
            closest_track = get_closest_track(tracks_name_ids.keys(),track_name)
            if closest_track != track_name:
                print '\twe approximate your song title:',track_name,'by:',closest_track
            preview = get_preview_from_trackid(tracks_name_ids[closest_track])
            if preview == '':
                print '\tsomething went wrong when looking by track id after artist id'
            else:
                return preview, tracks_name_ids[closest_track]
                
    # damn it! search by artist name + track title
    print "\tusing artist and title: %s by %s" % (track_name, artist_name)
    res = get_trackid_from_text_search(track_name,artistname=artist_name)
    if res is None:
        print '\tsomething went wrong when doing text search with artist and track name, no more ideas'
        return None, -1
    closest_track,trackid = res
    if closest_track != track_name:
        print '\twe approximate your song title:',track_name,'by:',closest_track
    preview = get_preview_from_trackid(trackid)
    if preview == '':
        print '\tsomething went wrong when looking by track id after text searching by artist and track name'
    else:
        return preview, trackid
    
    return None, -1
