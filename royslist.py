# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:53:21 2018

@author: Dell
"""

import urllib
from bs4 import BeautifulSoup
import re
import pandas as pd


url = "http://www.ucmp.berkeley.edu/arthropoda/crustacea/malacostraca/eumalacostraca/royslist/"
fp_index = urllib.request.urlopen(url)

index_html = fp_index.read()
index_soup = BeautifulSoup(index_html, "html5lib") # , "html5lib")
species_tags = index_soup.find_all('a', {'href': re.compile(r'species(.*)')})

p_php=re.compile("\".*\"")

df = pd.DataFrame
data_dict = {}

for index in range(len(species_tags)):
    tag = species_tags[index]
    
    m_php = p_php.findall(str(tag))
    m_php = m_php[0] # ["'species_link'"] -> "'species_link'"
    php_link = m_php[1:-1] # "'species_link'" -> "species_link"
    
    fp = urllib.request.urlopen((url + php_link))
    data_html = fp.read()
    data_soup = BeautifulSoup(data_html, "html5lib")
    # First, find species name
    table = data_soup.find("h2", {"align" : "center"}) # conveniently this seems to get it just right
    p_species_name = re.compile("<i>(.*)</i>") # p_* is pattern for something
    m_species_name = p_species_name.findall(str(table)) # m_* is match for something
    species_name = m_species_name[0]
    
    if 'Species' not in data_dict.keys():
        data_dict = {"Species" : [species_name]}
    else:
        data_dict["Species"].append(species_name)
    
    # Second, find species data
    paragraphs = data_soup.find_all('p')
    
    p_data = re.compile("<p><b>(.*)</b>(.*)</p>")
    keys=list(data_dict.keys()) # to keep track of which keys have been added to and which need fillers
    keys.remove("Species")
    
    for paragraph in paragraphs:
        paragraph = str(paragraph)
        m_data = p_data.match(paragraph)    
        
        if m_data: # if there is a match
            # first, separate key and data
            p_header = re.compile("<b>(.*)</b>")
            m_header = p_header.findall(paragraph)
            m_header = m_header[0]
            
            p_info = re.compile("</b>(.*)</p>")
            m_info = p_info.findall(paragraph)
            m_info = m_info[0]
                        
            if m_header not in data_dict.keys() and m_header is not "Aquarium Requirements:":
                # put into list, even is already a list
                # catch it up with n - 1 empty assignments
                temp = []
                for i in range(index - 1):
                    temp.append("")
                    
                data_dict[m_header] = temp
                data_dict[m_header].append([m_info])
            else:
                # if this doesn't work, step through. It's possible that we
                # have a mutability problem.
                data_dict[m_header].append([m_info])
                keys.remove(m_header)
            
    for header in keys: # all leftover keys are taken care of, appended
        # with empty assignment
        data_dict[header].append("")
                
pd.DataFrame(data_dict).to_csv(r'C:\Users\Dell\Google Drive\BYB\royslist.csv', index=False)


            
            
            
            