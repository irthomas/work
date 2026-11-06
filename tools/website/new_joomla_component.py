# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:40:42 2026

@author: iant

SET UP NEW JOOMLA 5 COMPONENT

RUN ON DEV FIRST

"""
import os

# ALLOW_OVERWRITE = True
ALLOW_OVERWRITE = False

DEV = True
# DEV = False

if DEV:
    COMPONENT_ROOT_PATH = r"W:\websites\dev\readonly\nomad\served\Joomla\components\com_nomad"
else:
    COMPONENT_ROOT_PATH = r"W:\websites\prod\readonly\nomad\served\Joomla\components\com_nomad"

# first letters capitalised with underscores between
component_name = "Forward_Model"


model_path = os.path.join(COMPONENT_ROOT_PATH, "models", "%s.php" % component_name.lower())


model_text = """
<?php
defined('_JEXEC') or die;
jimport('joomla.application.component.modellist');
jimport('joomla.application.component.helper');

class NomadModel%s extends JModelLegacy
{
    function getData()
    {
    return("0");
    }
}
""" % component_name

with open(model_path, "w") as f:
    f.write(model_text)

views_dir_path = os.path.join(COMPONENT_ROOT_PATH, "views", "%s" % component_name.lower())
os.makedirs(views_dir_path, exist_ok=ALLOW_OVERWRITE)

index_path = os.path.join(views_dir_path, "index.html")

index_text = """
<html><body bgcolor="#FFFFFF">
<h1>index.html</h1>
</body></html>
"""

with open(index_path, "w") as f:
    f.write(index_text)

view_php_path = os.path.join(views_dir_path, "view.html.php")

view_php_text = r"""
<?php
defined('_JEXEC') or die('Restricted access');
use Joomla\CMS\MVC\View\HtmlView;
use Joomla\CMS\Factory;

class NomadView%s  extends HtmlView
{
    public function display($tpl = null)
    {
        parent::display($tpl);
    }
}
""" % component_name

with open(view_php_path, "w") as f:
    f.write(view_php_text)

views_tmpl_dir_path = os.path.join(views_dir_path, "tmpl")
os.makedirs(views_tmpl_dir_path, exist_ok=ALLOW_OVERWRITE)

default_xml_path = os.path.join(views_tmpl_dir_path, "default.xml")

default_xml_text = """
<?xml version="1.0" encoding="utf-8"?>
<metadata>
    <layout title="COM_NOMAD_%s">
    <message><![CDATA[COM_NOMAD_%s]]></message>
    </layout>
</metadata>
""" % (component_name.upper(), component_name.upper())

with open(default_xml_path, "w") as f:
    f.write(default_xml_text)

default_php_path = os.path.join(views_tmpl_dir_path, "default.php")

default_php_text = r"""
<?php
// No direct access to this file
defined('_JEXEC') or die('Restricted access');
use Joomla\CMS\Form\Form;
use Joomla\CMS\Factory;

$url = 'https://wsgi.aeronomie.be/runner/nomad_opendap/name.h5.json?group.dset';
$json = file_get_contents($url);
$data = json_decode($json, true);
$var = array_column($data["group"]["dset"], 0);

var_dump($var);
?>
"""

with open(default_php_path, "w") as f:
    f.write(default_php_text)
