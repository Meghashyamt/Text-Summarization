import fitz
import os
import pandas as pd
import re
import tabula
import tabulate
import io
def pdf_parser(path,name):
    make_path = './output/'
    if not os.path.exists(os.path.join(make_path,os.path.splitext(name)[0])):
        os.mkdir(os.path.join(make_path, os.path.splitext(name)[0]))
        os.mkdir(os.path.join(make_path, os.path.splitext(name)[0]+'/signtures'))
        os.mkdir(os.path.join(make_path, os.path.splitext(name)[0]+'/tables'))
    dic = {}
    doc = fitz.open(path)  # open document
    # print("Total pages in contract : ",len(doc))
    # print("Total No. of signature in document : ",len(doc.getPageImageList(0)))
    
    print(" ")
    print("Extracting information from",os.path.splitext(name)[0])
    print(" ")
    """for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]

            # Save the image using a suitable format (e.g., PNG)
            image_filename = os.path.join(make_path, os.path.splitext(name)[0] + f"_Page{page_num + 1}_Image{img_index}.png")
            with open(image_filename, "wb") as image_file:
                image_file.write(image_data)"""
    dic["Total pages in contract"]= len(doc)
    #dic["Total No. of signature in document"]=len(doc.getPageImageList(0))
    #dic["Total No. of signature in document"] = len(doc.get_page_image_list(0))
    dic["Total No. of signature in document"] = len(doc.get_page_images(0))


    #dic["Total No. of signature in document"] = len(doc[0].get_page_image_list())
    total_images = len(doc[0].get_images(full=True))
    dic["Total No. of signature in document"] = total_images
    typelist = ["DISTRIBUTION AGREEMENT","CONSULTANCY AGREEMENT","SPEAKER AGREEMENT","Consultancy Agreement","SPONSORSHIP AGREEMENT"]
    code = [r"(U\d+[a-zA-Z]{2}\d+[a-zA-Z]{3}\d{6})",r"(MCI\b\s{1}.*\d{5,10})"]
    date = r"(\s+([a-zA-Z]+\s+)+)[0-9]+-[a-zA-Z]+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?"
    s_name = ["BAXALTA BIOSCIENCE INDIA PVT. LTD","Baxalta Bioscience India Private Limited","Baxalta Bioscience India Pvt. Ltd","Baxalta Bioscience India Pvt. Ltd."]
    s_address = [r"(6th Floor, Tower-C, Building No.8, DLF Cyber City,\nDLF Phase-II, Gurgaon-122 002, Haryana, India)",r"(6th Floor, Tower-C, Building No.8, DLF Cyber City, DLF\nPhase-II, Gurgaon-122 002, Haryana, India)"]
    d_name = ["SHIVSAI INTERNATIONAL PVT LTD","ROHIT ENTERPRIESE, A SOLE PROPRIETORSHIP","TECHNO MEDICAL SERVICES PRIVATE LIMITED., A SOLE PROPRIETORSHIP","VARDHMAN HEALTH SPECIALITIES PVT LTD","Dr. Shaleen Agarwal",
              "Dr Apurba Ghosh, Pediatrician","Dr Satish G Kulkarni","Dr. Parimal Shripad Lawate, Consultant Gastroenterologist","DR. SHRIKANT MUKEWAR, Consultant Gastroenterologist","Dr Rashna Dass Hazarika, Pediatrician",
              "THE HAEMOPHILIA SOCIETY, CALCUTTA CHAPTER",r"Indian Society of Inborn Errors of Metabolism"]
    d_address = [r"Shop No-28, A-4, DDA Triveni\nMarket , Pachim Vihar,New Delhi -110063","Shop no. 1&3, 1312 Sadashiv Peth Antarang Apt. Ajitha Cooperative Society Opposite Bharat Natya Mandir Pune 411030",
                 "Industrial Focal Point, Near Ajit Samachar, E- 243, Phase- 8B, Sector 90, Mohali, SAS Nagar, Punjab- 160062", "199/5, 7th Main Rd, behind The Bangalore Hospital, 2nd Block, Jayanagar, Bangalore, Karnataka 560011",
                 "Max Super Specialty Hospital, New Delhi 110017","FRIGE’s Institute of Human Genetics, FRIGE House, Jodhpur Village Road, Satellite, Ahmedabad, Gujarat, India- 380015", "SINGHABARI, KALIKAPUR, E.M. BYPASS, KOLKATA-700099",
                 "SINGHABARI, KALIKAPUR, E.M. BYPASS, KOLKATA-700099","Midas Multispecialty Hospitals Pvt. Ltd.,/nMidas Heights, 7, Central Bazaar Road, Ramdaspeth, Nagpur – 440010","Department of GI and Liver Disease, Jehangir/nHospital, Pune 411001",
                 "16, Swagat, plot 2A, sector 9A, Vashi, Navi Mumbai, 400703","Institute of Child Health, Park Circus ,Ballygunj , Kolkata 700017,West Bengal, India","Max Super Specialty Hospital, New Delhi 110017","NEMCARE Superspeciality Hospital, Guwahati, Assam, India",
                 "Department of GI and Liver Disease, Jehangir Hospital, Pune 411001",]
    dtype=re.search('|'.join(typelist), doc[0].get_text(),flags=0)
    if dtype is None:
        dtype = ""
    else:
        dtype =dtype.group(0)
    # print("Contract type : ",dtype)
    dic["Contract type"]=dtype


    dic["Contract type"]=dtype

    contacts = []
    for i in range(len(doc)):
        value = re.findall(r'^(?:(?:\+|0{0,2})(\s*[\-]\s*)?|[0]?)?[789]\d{9}|(\d[ -]?){10}\d$', doc[i].get_text('text'), flags=0)
        if value !=[]:
            contacts.append(value)
        else:
            continue
    mails = []
    for i in range(len(doc)):
        value = re.findall(r'[a-zA-Z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', doc[i].get_text('text'), flags=0)
        if value !=[]:
            mails.append(value[0])
        else:
            continue

    SupplierName= re.search('|'.join(s_name), doc[0].get_text(), flags=re.MULTILINE)
    if SupplierName is None:
        SupplierName =""
    else:
        SupplierName =SupplierName.group(0)

    SupplierAddress = re.search("|".join(s_address), doc[0].get_text(), flags=re.MULTILINE)
    if SupplierAddress is None:
        SupplierAddress = ""
    else:
        SupplierAddress =SupplierAddress.group(0)
    DistributorName = re.search('|'.join(d_name), doc[0].get_text(), flags=re.MULTILINE)
    if DistributorName is None:
        DistributorName = ""
    else:
        DistributorName =DistributorName.group(0)
    DistributorAddress = re.search('|'.join(d_address), doc[0].get_text(), flags=re.MULTILINE)
    if DistributorAddress is None:
        DistributorAddress =""
    else:
        DistributorAddress =DistributorAddress.group(0)
    DistributorCIN_MCI = re.search('|'.join(code), doc[0].get_text())
    if DistributorCIN_MCI is None:
        DistributorCIN_MCI = ""
    else:
        DistributorCIN_MCI =DistributorCIN_MCI.group(0)
    EffectiveDate =re.search(date, doc[0].get_text()+doc[1].get_text())
    if EffectiveDate is None:
        EffectiveDate = ""
    else:
        EffectiveDate = EffectiveDate.group(0)


    # print("Supplier Name : ", SupplierName)
    # print("Supplier Address : ",SupplierAddress)
    # print("Distributor Name : ",DistributorName)
    # print("Distributor Address : ",DistributorAddress)
    # print("Distributor CIN or MCI : ",DistributorCIN_MCI)
    # print("Effective Date : ",EffectiveDate)
    # print("All Contacts in contract : ", contacts)
    #
    # print("All email ID in contract : ", mails)

    dic["Supplier Name"]= SupplierName
    dic["Supplier Address"]= SupplierAddress
    dic["Distributor Name"]= DistributorName
    dic["Distributor Address"]= DistributorAddress
    dic["Distributor CIN or MCI"]= DistributorCIN_MCI
    dic["Effective Date"] = EffectiveDate
    dic["All Contacts in contract"] = contacts
    dic["All email ID in contract"] = mails
    df = pd.DataFrame(dic.items(), columns=["Field Name", "Field Value"])
    df.to_csv(os.path.join("./output",os.path.splitext(name)[0] +"/"+name+'.csv'))
    for img in doc.getPageImageList(0):
        xref = img[0]
        pix = fitz.Pixmap(doc,xref)
        if pix.n<5:
            pix.writePNG(os.path.join(make_path, os.path.splitext(name)[0]+'/signtures/'+name+"P-%s.png"%(xref)))
        else:
            pix1 = fitz.Pixmap(fitz.csRGB,pix)
            pix1.writePNG(os.path.join(make_path, os.path.splitext(name)[0]+'/signtures/'+name+"P-%s.png"%(xref)))
            pix1= None
        pix=None
    # get all the tables from the pdf file.
    if dtype.lower()=="DISTRIBUTION AGREEMENT".lower():
        page =[28,30,33,34,37,38,40]
    elif dtype.lower()=="CONSULTANCY AGREEMENT".lower() or "SPEAKER AGREEMENT".lower():
        page =[4,5,6,11]
    else:
        page = [1,2,8]
    # tables = tabula.read_pdf(path, multiple_tables = True,pages=page)
    # iterate over extracted tables and export as excel individually
    # for i, table in enumerate(tables, start=1):
    #     table.to_excel(os.path.join(make_path, os.path.splitext(name)[0]+'/tables/' + f"table_{i}.xlsx"), index=False)
    # Save the final result as excel file
    tabula.convert_into(path,os.path.join(make_path, os.path.splitext(name)[0]+'/tables/' +os.path.splitext(name)[0] +".csv"),output_format="csv",pages=page)
    print("Done")
    print(" ")

    return dic